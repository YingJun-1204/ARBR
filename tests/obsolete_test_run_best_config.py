import unittest
import sys
import os
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_best_config import parse_pred_len_from_filename


class TestRunBestConfig(unittest.TestCase):
    def test_parse_pred_len_from_filename(self):
        self.assertEqual(parse_pred_len_from_filename("best_etth1_router_96.json"), 96)
        self.assertEqual(parse_pred_len_from_filename("best_etth2_router_192.json"), 192)
        self.assertEqual(parse_pred_len_from_filename("some_other_file.json"), 96)

    def test_horizon_path_replacement(self):
        config_path = "loss\u65e5\u5fd7/best_etth1_router_96.json"
        horizon = 192
        dir_name, file_name = os.path.split(config_path)
        new_file_name = re.sub(r"_(\d+)\.json$", f"_{horizon}.json", file_name)
        new_path = os.path.join(dir_name, new_file_name)
        self.assertEqual(new_path, "loss\u65e5\u5fd7\\best_etth1_router_192.json" if os.name == 'nt' else "loss\u65e5\u5fd7/best_etth1_router_192.json")

    def test_result_json_update_logic(self):
        # 模拟结果写入与更新逻辑
        temp_result_file = "test_result_temp_horizon.json"
        if os.path.exists(temp_result_file):
            os.remove(temp_result_file)
            
        horizons = ["96", "192", "336", "720"]
        
        # 1. 模拟第一次跑 horizon 96 (文件不存在时初始化)
        pred_len = 96
        last_mse = 0.35
        last_mae = 0.38
        
        if os.path.exists(temp_result_file):
            with open(temp_result_file, "r") as rf:
                results_data = json.load(rf)
        else:
            results_data = {}
            
        for h in horizons:
            if h not in results_data:
                results_data[h] = None
                
        results_data[str(pred_len)] = {
            "mse": last_mse,
            "mae": last_mae
        }
        
        with open(temp_result_file, "w") as wf:
            json.dump(results_data, wf, indent=4)
            
        # 验证 96 的值已被填入，其他是 None
        with open(temp_result_file, "r") as rf:
            data = json.load(rf)
        self.assertEqual(data["96"]["mse"], 0.35)
        self.assertIsNone(data["192"])
        self.assertIsNone(data["336"])
        self.assertIsNone(data["720"])
        
        # 2. 模拟第二次跑 horizon 192 (增量更新)
        pred_len = 192
        last_mse_2 = 0.40
        last_mae_2 = 0.45
        
        with open(temp_result_file, "r") as rf:
            results_data = json.load(rf)
            
        results_data[str(pred_len)] = {
            "mse": last_mse_2,
            "mae": last_mae_2
        }
        
        with open(temp_result_file, "w") as wf:
            json.dump(results_data, wf, indent=4)
            
        # 验证 96 和 192 的值都存在，其他依旧是 None
        with open(temp_result_file, "r") as rf:
            data = json.load(rf)
        self.assertEqual(data["96"]["mse"], 0.35)
        self.assertEqual(data["192"]["mse"], 0.40)
        self.assertIsNone(data["336"])
        
        # 清理
        if os.path.exists(temp_result_file):
            os.remove(temp_result_file)


if __name__ == "__main__":
    unittest.main()
