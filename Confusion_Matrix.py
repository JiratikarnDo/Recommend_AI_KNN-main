# ข้อมูลที่ได้จาก Rule-Based และ KNN
rule_based = {132, 131, 139, 151, 152, 168, 169, 174}  # ข้อมูลจริง
knn = {132, 151, 152, 168, 174, 125, 131, 133}        # ข้อมูลที่ KNN ทำนาย

# ผู้ใช้ทั้งหมดในระบบ 50 คน (UserID 1-50)
all_users = set(range(1, 51))  # ผู้ใช้ทั้งหมดในระบบ

# หา True Positive (TP): อยู่ทั้งใน KNN และ Rule-Based
tp = rule_based.intersection(knn)

# หา False Positive (FP): อยู่ใน KNN แต่ไม่อยู่ใน Rule-Based
fp = knn.difference(rule_based)

# หา False Negative (FN): อยู่ใน Rule-Based แต่ไม่อยู่ใน KNN
fn = rule_based.difference(knn)

# รวมผู้ใช้ที่ถูกแนะนำ (TP + FP + FN)
users_recommended = tp.union(fp).union(fn)

# หา True Negative (TN): ไม่อยู่ทั้งใน Rule-Based และ KNN
tn = all_users.difference(users_recommended)


# แสดงผลลัพธ์ Confusion Matrix
print("Confusion Matrix:")
print(f"True Positives (TP): {len(tp)} --> {tp}")
print(f"False Positives (FP): {len(fp)} --> {fp}")
print(f"False Negatives (FN): {len(fn)} --> {fn}")
print(f"True Negatives (TN): {len(tn)} --> {tn}")

# สรุปผลรวม
print("\nSummary:")
print(f"Total Users: {len(all_users)}")
print(f"TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}, TN: {len(tn)}")
