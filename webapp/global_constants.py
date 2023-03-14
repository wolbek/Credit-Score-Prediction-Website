grade = ["A", "B", "C", "D", "E","F","G"]
home_ownership = ["RENT_OTHER_NONE_ANY", "OWN", "MORTGAGE"]
addr_state = ["ND_NE_IA_NV_FL_HI_AL", "NM_VA", "NY", "OK_TN_MO_LA_MD_NC", "CA", "UT_KY_AZ_NJ", "AR_MI_PA_OH_MN", "RI_MA_DE_SD_IN", "GA_WA_OR", "WI_MT", "TX", "IL_CT", "KS_SC_CO_VT_AK_MS", "WV_NH_WY_DC_ME_ID"]
verification_status = ["Verified", "Not Verified","Source Verified"]
purpose = ["educ__sm_b__wedd__ren_en__mov__house", "credit_card", "debt_consolidation", "oth__med__vacation", "major_purch__car__home_impr"]
initial_list_status = ["f","w"]
term = ["36","60"]
emp_length = ["0", "1", "2-4", "5-6", "7-9", "10"]
mths_since_issue_d = ["<38", "38-39", "40-41", "42-48", "49-52", "53-64", "65-84", ">84"]
int_rate = ["<9.548", "9.548-12.025", "12.025-15.74", "15.74-20.281", ">20.281"]
mths_since_earliest_cr_line = ["<140", "141-164", "165-247", "248-270", "271-352", ">352"]
inq_last_6mths = ["0","1-2","3-6",">6"]
acc_now_delinq = ["0",">=1"]
annual_inc = ["<20K", "20K-30K", "30K-40K", "40K-50K", "50K-60K", "60K-70K", "70K-80K", "80K-90K", "90K-100K", "100K-120K", "120K-140K", ">140K"]
dti = ["<=1.4", "1.4-3.5", "3.5-7.7", "7.7-10.5", "10.5-16.1", "16.1-20.3", "20.3-21.7", "21.7-22.4", "22.4-35", ">35"]
mths_since_last_delinq = ["Missing", "0-3", "4-30", "31-56", ">=57"]
mths_since_last_record = ["Missing", "0-2", "3-20", "21-31", "32-80", "81-86", ">86"]

# Independent variables considered:
# grade
# home_ownership
# addr_state
# verification_status
# purpose
# initial_list_status
# term
# emp_length
# mths_since_issue_d 
# int_rate 
# mths_since_earliest_cr_line
# inq_last_6mths
# acc_now_delinq 
# annual_inc
# dti
# mths_since_last_delinq
# mths_since_last_record