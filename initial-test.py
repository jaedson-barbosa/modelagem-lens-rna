from readdata import *
from model import *
from plotdata import *

data = read_test_data('initial')
data = concat_delayed_flows(data)
(x, y) = pd2dataarray(data)

model = load_model("initial")
(MAPE, MSE, y_pred) = check_results(x, y, model)
print(f"MAPE: {MAPE}%\nMSE: {MSE}")

plot_name = 'initial-test'
plot_2_lines(y_pred[:, 0], 'FT_1A estimated', y[:, 0], 'FT_1A real', plot_name + '-1A')
plot_2_lines(y_pred[:, 1], 'FT_3A estimated', y[:, 1], 'FT_3A real', plot_name + '-3A')
