// stock.cpp : Defines the entry point for the application.
//

#include "stock.h"

using namespace std;

#include "../../miapi/miapi/std/csv_syntax.h"
#include "../../miapi/miapi/std/math.h"
#include "../../miapi/miapi/std/mem.h"
#include "../../miapi/miapi/std/nn.h"
#include "../../miapi/miapi/std/utf.h"
using namespace miapi;

//SOTCK_ACTION
namespace SOTCK_ACTION
{
	enum
	{
		NONE,
		BUY,
		SELL,

		UNKNOWN
	};
}

int main()
{
	string filename = "output/2317.TW.csv";

	double money = 1000;
	double used = 0;
	double count = 0;

	double count_max = 1;

	//load file
	auto file = mem::load_file<utf::utf8>(filename);
	if (file.first == nullptr)
		return __LINE__;

	csv_syntax<utf::utf8*> csv(",\t");
	decltype(csv)::data_type cd;

	if (!csv.read(cd, utf::skipbom(file.first.get()), mem::offset(file.first.get(), file.second)))
		return __LINE__;

	//remove useless data
	int r = static_cast<int>(cd.size());
	int c = static_cast<int>(cd[0].size());

	for (size_t i = 1; i < cd.size();)
	{
		if (0 == utf::wton<double>(cd[i][c - 1]).first)
		{
			cd.erase(cd.begin() + i);
		}
		else
		{
			++i;
		}
	}

	//convert data
	r = static_cast<int>(cd.size() - 1);
	c = static_cast<int>(cd[0].size() - 1);

	auto data_buffer = mem::allocate<double>(r * c * sizeof(double));
	auto data = math::dimension<double*>{ data_buffer.get(), (int)c };

	for (int i = 1; i < cd.size(); ++i)
	{
		for (int j = 1; j < cd[0].size(); ++j)
		{
			data[i - 1][j - 1] = utf::wton<double>(cd[i][j]).first;
		}
	}

	math::matrix_print(cout, data, 10, c, 2);
	cout << endl;

	//check data range
	double price_min = data[0][0];
	double price_max = data[0][0];
	double vol_min = data[0][c - 1];
	double vol_max = data[0][c - 1];

	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < c - 1; ++j)
		{
			double p = data[i][j];

			if (price_min > p)
				price_min = p;

			if (price_max < p)
				price_max = p;
		}

		double v = data[i][c - 1];

		if (vol_min > v)
			vol_min = v;

		if (vol_max < v)
			vol_max = v;
	}

	if (0 == price_min)
		return __LINE__;

	double price_div = sqrt(price_min * price_min + price_max * price_max);
	double vol_div = sqrt(vol_min * vol_min + vol_max * vol_max);
	double money_div = count_max * price_div + money;

	cout << "price_div : " << price_min << ", " << price_max << ", " << price_div << endl;
	cout << "vol_div : " << vol_min << ", " << vol_max << ", " << vol_div << endl;
	cout << "money_div : " << money_div << endl;
	cout << endl;

	//normalize data
	auto ndata_buffer = mem::allocate<double>(r * c * sizeof(double));
	auto ndata = math::dimension<double*>{ ndata_buffer.get(), (int)c };

	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < c - 1; ++j)
			ndata[i][j] = data[i][j] / price_div;

		ndata[i][c - 1] = data[i][c - 1] / vol_div;
	}

	math::matrix_print(cout, ndata, 10, c);
	cout << endl;

	//dqn
	double tmp_money = money;
	double tmp_used = used;
	double tmp_count = count;

	int32_t retry = 10;

	const int32_t epoch = 100;

	const double fee_buy = 0.15 / 100.0;
	const double fee_sell = (0.15 + 0.3) / 100.0;

	const double rate = 0.1;
	const double gamma = 0.9;
	const int32_t window = 30;

	const int32_t test = 30 + window;

	//
	int32_t in_size = window * c + 2;
	int32_t out_size = SOTCK_ACTION::UNKNOWN;
	int32_t inner_size = in_size * out_size;

	//nwt
	nn::network<double> nwt;
	nwt.layout(0, in_size + 1, inner_size, new nn::act_tanh<double>(), new nn::opt_adam<double>());
	nwt.layout(1, inner_size + 1, out_size, new nn::act_elu<double>(), new nn::opt_adam<double>());
	nwt.connect();
	nwt.io_reset(1);

	//nw
	nn::network<double> nw;

	//target
	auto* target = nw.io_ptr(static_cast<int32_t>(nwt.io_size()), nwt.out_size(), 0);

	nw.layout(0, in_size + 1, inner_size, new nn::act_tanh<double>(), new nn::opt_adam<double>());
	nw.layout(1, inner_size + 1, out_size, new nn::act_elu<double>(), new nn::opt_adam<double>());
	nw.connect();
	nw.io_reset(1);

	//random
	nw.neural_reset(0.05 / inner_size, 0.1 / inner_size);

	std::uniform_int_distribution<> dist_greedy(0, r - test);
	std::uniform_int_distribution<> dist_action(SOTCK_ACTION::BUY, SOTCK_ACTION::SELL);

	//run
	std::shared_ptr<double> work = nullptr;
	std::default_random_engine re(utility::clock_to_time<uint32_t>());

	for (int32_t e = 0; e < epoch;)
	{
		for (auto i = window; i < r - test; ++i)
		{
			int32_t state = i - window;

			math::vector_assign(nw.in(), ndata[state], nw.in_size() - 2);
			nw.in()[nw.in_size() - 2] = tmp_money / money_div;
			nw.in()[nw.in_size() - 1] = tmp_count / count_max;

			//nwt
			if (0 == i % 10)
			{
				for (auto j = 0; j < nw.neural_size(); ++j)
					math::matrix_assign(nwt[j].weight(), nw[j].weight(), nw[j].row(), nw[j].column());
			}

			nw.forward(true);

			//action
			math::vector_assign(target, nw.out(), nw.out_size());
			int32_t action = math::vector_max_index(target, nw.out_size());
			if (dist_greedy(re) >= i)
				action = dist_action(re);

			//reward
			double reward = -0.5;

			double close_today = data[i - 1][3];
			double close_next = data[i][3];
			double high_next = data[i][1];
			double low_next = data[i][2];

			switch (action)
			{
			case SOTCK_ACTION::NONE:
				if (tmp_count > 0)
				{
					double average = tmp_used / tmp_count;
					reward = (close_today - average) / average - fee_sell;
				}
				else
				{
					reward = (close_next - close_today) / close_today + fee_buy + fee_sell;
				}

				reward = std::min(-reward, 0.0) * 100.0;
				break;
			case SOTCK_ACTION::BUY:
				if (tmp_money > close_today && tmp_count < count_max)
				{
					reward = (close_next - close_today) / close_today + fee_buy + fee_sell;

					if (close_today >= low_next)
					{
						double v = close_today * (1.0 + fee_buy);
						tmp_money -= v;
						tmp_used += v;
						++tmp_count;
					}
				}
				break;
			case SOTCK_ACTION::SELL:
				if (tmp_count > 0)
				{
					double average = tmp_used / tmp_count;
					reward = (close_today - average) / average - fee_sell;

					if (close_today <= high_next)
					{
						double v = close_today * (1.0 - fee_sell);
						tmp_money += v;
						tmp_used -= v;
						--tmp_count;
					}
				}
				break;
			default:
				return __LINE__;
			}

			math::vector_assign(nwt.in(), ndata[state + 1], nwt.in_size() - 2);
			nwt.in()[nwt.in_size() - 2] = tmp_money / money_div;
			nwt.in()[nwt.in_size() - 1] = tmp_count / count_max;

			nwt.forward(false);

			target[action] = reward * gamma + rate * math::vector_max(nwt.out(), nwt.out_size());

			//backward
			for (int32_t j = 0; j < 4; ++j)
			{
				work = nw.backward(target, work);
				nw.forward(true);
			}
		}

		cout << ++e << " / " << epoch << " (" << retry - 1 << ")" << endl;

		if (0 == e % 10)
		{
			math::vector_assign(nw.in(), ndata[r - test - window], nw.in_size() - 2);
			nw.in()[nw.in_size() - 2] = money / money_div;
			nw.in()[nw.in_size() - 1] = count / count_max;

			nw.forward(false);

			if (math::vector_max(nw.out(), nw.out_size()) < -0.999 || nw.out()[0] != nw.out()[0])
			{
				if (--retry < 1)
					return __LINE__;

				nw.neural_reset(0.05 / inner_size, 0.1 / inner_size);
				e = 0;
			}
		}
	}
	cout << endl;

	//test
	tmp_money = money;
	tmp_used = used;
	tmp_count = count;

	for (auto i = r - test + window; i < r; ++i)
	{
		int32_t state = i - window;

		math::vector_assign(nw.in(), ndata[state], nw.in_size() - 2);
		nw.in()[nw.in_size() - 2] = tmp_money / money_div;
		nw.in()[nw.in_size() - 1] = tmp_count / count_max;

		nw.forward(false);
		int32_t action = math::vector_max_index(nw.out(), nw.out_size());

		nw.out_print(cout);

		//reward
		double close_today = data[i - 1][3];
		double close_next = data[i][3];
		double high_next = data[i][1];
		double low_next = data[i][2];

		switch (action)
		{
		case SOTCK_ACTION::NONE:
			break;
		case SOTCK_ACTION::BUY:
			if (tmp_money > close_today && tmp_count < count_max)
			{
				if (close_today >= low_next)
				{
					double v = close_today * (1.0 + fee_buy);
					tmp_money -= v;
					tmp_used += v;
					++tmp_count;

					cout << cd[i][0] << " -" << close_today << " = " << tmp_money << endl;
				}
			}
			break;
		case SOTCK_ACTION::SELL:
			if (tmp_count > 0)
			{
				if (close_today <= high_next)
				{
					double v = close_today * (1.0 - fee_sell);
					tmp_money += close_today;
					tmp_used -= close_today;
					--tmp_count;

					cout << cd[i][0] << " +" << close_today << " = " << tmp_money << endl;
				}
			}
			break;
		default:
			return __LINE__;
		}
	}
	cout << endl;

	cout << "money : " << money << " -> " << tmp_money << " %" << (tmp_money - money) / money << endl;
	cout << "used : " << used << " -> " << tmp_used << endl;
	cout << "count : " << count << " -> " << tmp_count << endl;
	cout << endl;

	return 0;
}
