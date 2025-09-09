#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <functional>
#include <windows.h>
#include <map>

constexpr int GRID_X1 = 747;
constexpr int GRID_Y1 = 276;
constexpr int GRID_X2 = 1301;
constexpr int GRID_Y2 = 832;

constexpr int BLOCK_SIZE = 90;
constexpr int SPACING = 3;
constexpr int GRID_SIZE = 6;

struct shape_data_t
{
	std::string name;
	int vertices;
	double perimeter;
	double area;
	double circularity;
};

size_t board_hash(const std::vector<std::vector<int>>& board)
{
	std::hash<std::string> hasher;
	std::string flat;
	for (auto& row : board) for (int cell : row) flat.push_back(static_cast<char>(cell + 50));
	return hasher(flat);
}

shape_data_t identify_shape(const std::vector<cv::Point>& hull)
{
	double peri = cv::arcLength(hull, true);
	double area = cv::contourArea(hull);

	if (peri == 0) return { "unknown", 0, 0.0, 0.0, 0.0 };

	std::vector<cv::Point> approx;
	cv::approxPolyDP(hull, approx, 0.05 * peri, true);
	int vertices = static_cast<int>(approx.size());

	double circularity = 4 * CV_PI * area / (peri * peri);
	std::string shape = "unknown";

	if (circularity > 0.9) shape = "circle";
	else if (vertices == 3) shape = "triangle";
	else if (vertices == 4)
	{
		std::vector<double> edges;
		for (int i = 0; i < 4; i++)
		{
			cv::Point2f p1 = approx[i];
			cv::Point2f p2 = approx[(i + 1) % 4];
			edges.push_back(cv::norm(p1 - p2));
		}
		double ratio = *std::max_element(edges.begin(), edges.end()) / *std::min_element(edges.begin(), edges.end());
		shape = (ratio < 1.2) ? "rhombus" : "nonstnd";
	}
	else if (vertices >= 5) shape = "nonstnd";

	return { shape, vertices, std::round(peri * 100) / 100.0, std::round(area * 100) / 100.0,std::round(circularity * 100) / 100.0 };
}

cv::Mat screenshot(int x, int y, int width, int height)
{
	HDC hScreen = GetDC(NULL);
	HDC hDC = CreateCompatibleDC(hScreen);
	HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, width, height);
	SelectObject(hDC, hBitmap);

	BitBlt(hDC, 0, 0, width, height, hScreen, x, y, SRCCOPY | CAPTUREBLT);

	BITMAPINFOHEADER bi;
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = width;
	bi.biHeight = -height;
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	cv::Mat mat(height, width, CV_8UC4);
	GetDIBits(hDC, hBitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

	ReleaseDC(NULL, hScreen);
	DeleteDC(hDC);
	DeleteObject(hBitmap);

	cv::Mat bgr;
	cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
	return bgr;
}

struct move_result_t
{
	std::pair<int, int> first;
	std::pair<int, int> second;
	int score;
};

std::pair<std::vector<std::vector<bool>>, int> find_matches(const std::vector<std::vector<int>>& grid)
{
	std::vector<std::vector<bool>> matched(GRID_SIZE, std::vector<bool>(GRID_SIZE, false));

	for (int r = 0; r < GRID_SIZE; r++)
	{
		int c = 0;
		while (c < GRID_SIZE - 2)
		{
			int val = grid[r][c];
			if (val != -1 && val == grid[r][c + 1] && val == grid[r][c + 2])
			{
				int k = c;
				while (k < GRID_SIZE && grid[r][k] == val)
				{
					matched[r][k] = true;
					k++;
				}
				c = k;
			}
			else c++;
		}
	}

	for (int c = 0; c < GRID_SIZE; c++)
	{
		int r = 0;
		while (r < GRID_SIZE - 2)
		{
			int val = grid[r][c];
			if (val != -1 && val == grid[r + 1][c] && val == grid[r + 2][c])
			{
				int k = r;
				while (k < GRID_SIZE && grid[k][c] == val)
				{
					matched[k][c] = true;
					k++;
				}
				r = k;
			}
			else r++;
		}
	}

	int total = 0;
	for (int r = 0; r < GRID_SIZE; r++) for (int c = 0; c < GRID_SIZE; c++) if (matched[r][c]) total++;

	return { matched, total };
}

void apply_gravity_no_refill(std::vector<std::vector<int>>& grid)
{
	for (int c = 0; c < GRID_SIZE; c++)
	{
		int write_row = GRID_SIZE - 1;
		for (int r = GRID_SIZE - 1; r >= 0; r--)
		{
			if (grid[r][c] != -1)
			{
				grid[write_row][c] = grid[r][c];
				write_row--;
			}
		}
		for (int r = write_row; r >= 0; r--) grid[r][c] = -1;
	}
}

int simulate_once_no_random(std::vector<std::vector<int>>& grid)
{
	auto [matched, gained] = find_matches(grid);
	if (gained == 0) return 0;

	for (int r = 0; r < GRID_SIZE; r++)
		for (int c = 0; c < GRID_SIZE; c++)
			if (matched[r][c]) grid[r][c] = -1;

	apply_gravity_no_refill(grid);
	return gained;
}

move_result_t best_move(const std::vector<std::vector<int>>& board)
{
	int best_score = -1;
	std::pair<int, int> best_a = { -1,-1 }, best_b = { -1,-1 };

	std::vector<std::pair<int, int>> dirs = { {0,1},{1,0} };

	for (int r = 0; r < GRID_SIZE; r++)
	{
		for (int c = 0; c < GRID_SIZE; c++)
		{
			for (auto [dr, dc] : dirs)
			{
				int nr = r + dr, nc = c + dc;

				if (nr < GRID_SIZE && nc < GRID_SIZE)
				{
					auto grid_copy = board;
					std::swap(grid_copy[r][c], grid_copy[nr][nc]);
					int score = simulate_once_no_random(grid_copy);

					if (score > best_score ||
						(score == best_score && best_a.first != -1 &&
							std::make_pair(std::make_pair(r, c), std::make_pair(nr, nc)) <
							std::make_pair(best_a, best_b)))
					{
						best_score = score;
						best_a = { r,c };
						best_b = { nr,nc };
					}
				}
			}
		}
	}

	if (best_score < 0) best_score = 0;
	return { best_a,best_b,best_score };
}

int main()
{
	MoveWindow(GetConsoleWindow(), 0, 0, 500, 400, TRUE);
	Sleep(200);

	int width = GRID_X2 - GRID_X1;
	int height = GRID_Y2 - GRID_Y1;

	cv::Mat img = screenshot(GRID_X1, GRID_Y1, width, height);
	if (img.empty()) return 0;

	std::map<std::string, int> SHAPE_TO_COLOR
	{
		{"circle", 0},
		{"triangle", 1},
		{"rhombus", 2},
		{"nonstnd", 3},
		{"unknown", -1}
	};

	std::vector<std::vector<int>> board(GRID_SIZE, std::vector<int>(GRID_SIZE, -1));

	for (int row = 0; row < GRID_SIZE; row++)
	{
		for (int col = 0; col < GRID_SIZE; col++)
		{
			int x1 = col * (BLOCK_SIZE + SPACING);
			int y1 = row * (BLOCK_SIZE + SPACING);
			int bw = std::min(BLOCK_SIZE, img.cols - x1);
			int bh = std::min(BLOCK_SIZE, img.rows - y1);
			if (bw <= 0 || bh <= 0) continue;

			cv::Rect roi(x1, y1, bw, bh);
			cv::Mat block = img(roi);

			cv::Mat gray;
			cv::cvtColor(block, gray, cv::COLOR_BGR2GRAY);
			cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

			cv::Mat edges;
			cv::Canny(gray, edges, 50, 150);
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			cv::dilate(edges, edges, kernel);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			if (!contours.empty())
			{
				std::vector<cv::Point> merged;
				for (auto& c : contours) merged.insert(merged.end(), c.begin(), c.end());
				std::vector<cv::Point> hull;
				cv::convexHull(merged, hull);
				shape_data_t data = identify_shape(hull);
				int color_val = SHAPE_TO_COLOR.count(data.name) ? SHAPE_TO_COLOR[data.name] : -1;
				board[row][col] = color_val;
			}
		}
	}

	printf("Board hash: %llu\n", board_hash(board));

	for (auto& row : board)
	{
		printf("[");
		for (size_t i = 0; i < row.size(); i++)
		{
			printf("%d", row[i]);
			if (i + 1 < row.size()) printf(", ");
		}
		printf("]\n");
	}

	move_result_t move = best_move(board);

	if (move.score > 0)
	{
		printf("Best move: (Row %d, Col %d) <-> (Row %d, Col %d) | Score: %d\n",
			move.first.first + 1, move.first.second + 1, move.second.first + 1, move.second.second + 1, move.score);

		cv::Mat dimg = img.clone();

		cv::Point pt1(
			move.first.second * (BLOCK_SIZE + SPACING) + BLOCK_SIZE / 2,
			move.first.first * (BLOCK_SIZE + SPACING) + BLOCK_SIZE / 2
		);

		cv::Point pt2(
			move.second.second * (BLOCK_SIZE + SPACING) + BLOCK_SIZE / 2,
			move.second.first * (BLOCK_SIZE + SPACING) + BLOCK_SIZE / 2
		);

		cv::arrowedLine(dimg, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA, 0, 0.1);

		cv::circle(dimg, pt1, 10, cv::Scalar(255, 0, 0), 2);
		cv::circle(dimg, pt2, 10, cv::Scalar(255, 0, 0), 2);

		std::string score_text = "Score: " + std::to_string(move.score);
		cv::putText(dimg, score_text, cv::Point(10, 30),
			cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

		cv::imshow("If294378tfuiof", dimg);

		while (true)
		{
			int key = cv::waitKey(0) & 0xFF;
			if (key == 'q' || key == 'Q')
			{
				break;
			}
		}
	}

	return 0;
}
