import sqlite3

conn = sqlite3.connect("loss_mdagV12/best_ettm2_router_96.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM studies")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
