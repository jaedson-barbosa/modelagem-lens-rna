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
plot_n_lines([
    (y_pred[:, 0], 'Estimated'),
    (y[:, 0], 'Measured')
], [(x[:, 2], 'Frequency')], plot_name + '-1A')
plot_n_lines([
    (y_pred[:, 1], 'Estimated'),
    (y[:, 1], 'Measured')
], [(x[:, 2], 'Frequency')], plot_name + '-3A')
