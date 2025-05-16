import csv
from datetime import datetime

def mark_attendance(name):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('database/attendence.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])
    print(f"✅ Marked attendance for {name}")
