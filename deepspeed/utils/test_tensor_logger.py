from tensor_logger import TensorLogger

logger0 = TensorLogger()
print(f"logger0.tensor_dict = {logger0.tensor_dict}")

logger0.add_tensor(tensor=5, tensor_id=0, group_id=0)
logger0.add_tensor(tensor=10, tensor_id=1, group_id=0)
logger0.add_tensor(tensor=15, tensor_id=0, group_id=1)
logger0.add_tensor(tensor=25, tensor_id=1, group_id=1)
logger0.add_tensor(tensor=30, tensor_id=0, group_id=2)
logger0.add_tensor(tensor=35, tensor_id=1, group_id=2)

print(f"logger0.tensor_dict = {logger0.tensor_dict}")

logger1 = TensorLogger()

print(f"logger1.tensor_dict = {logger1.tensor_dict}")

logger1.add_tensor(tensor=5, tensor_id=0, group_id=0)
logger1.add_tensor(tensor=15, tensor_id=1, group_id=0)
logger1.add_tensor(tensor=15, tensor_id=0, group_id=1)
logger1.add_tensor(tensor=30, tensor_id=1, group_id=1)
logger1.add_tensor(tensor=30, tensor_id=0, group_id=2)
logger1.add_tensor(tensor=40, tensor_id=1, group_id=2)

print(f"logger1.tensor_dict = {logger1.tensor_dict}")

#def compare_loggers(logger0, logger1):
