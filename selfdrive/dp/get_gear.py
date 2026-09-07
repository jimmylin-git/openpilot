import glob
from openpilot.tools.lib.logreader import LogReader

# 找出所有 realdata 裡面的 rlog.zst 檔案
rlogs = glob.glob('/data/media/0/realdata/*/rlog.zst')

if not rlogs:
    print("找不到任何 rlog.zst 檔案！")
else:
    # 讀取最新的幾個檔案，或者指定檢查最新的這一檔
    latest_rlog = sorted(rlogs)[-1]
    print(f"正在深度解析記錄檔: {latest_rlog}\n")

    count = 0
    gear_changes = []

    try:
        for msg in LogReader(latest_rlog):
            if msg.which() == 'carState':
                gear = msg.carState.gearShifter
                # 記錄所有出現過的狀態及其發生次數
                gear_changes.append(gear)
                count += 1

        print(f"總共掃描到 {count} 筆 carState 資料")

        # 看看有沒有包含其他可能的自定義數值或原始 CAN 數值
        # 讓我們印出前幾筆或統計一下各個檔位出現的頻率
        from collections import Counter
        print("檔位出現次數統計:", Counter(gear_changes))

    except Exception as e:
        print(f"解析發生錯誤: {e}")