from readdata import *
from model import *
from plotdata import *

files = ['30hz', '35hz', '40hz', '45hz', '50hz']
data = pd.concat((read_training_data(f, True) for f in files))
data = concat_delayed_flows(data)
(x, y) = pd2dataarray(data, True)

model = train_model(x, y, 4, [4], 100, "complete")
(MAPE, MSE, y_pred) = check_results(x, y, model)
print(f"MAPE: {MAPE}%\nMSE: {MSE}")

plot_name = 'complete-training'
plot_2_lines(y_pred[:, 0], 'FT_1A estimated', y[:, 0], 'FT_1A real', plot_name + '-1A')
plot_2_lines(y_pred[:, 1], 'FT_3A estimated', y[:, 1], 'FT_3A real', plot_name + '-3A')
