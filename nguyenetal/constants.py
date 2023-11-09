"""Constant variables for Nguyen et al. notebooks."""

# PPMI file names
FILENAME_DEMOGRAPHICS = "Demographics.csv"
FILENAME_PD_HISTORY = "PD_Diagnosis_History.csv"
FILENAME_AGE = "Age_at_visit.csv"
FILENAME_SOCIO = "Socio-Economics.csv"
FILENAME_PARTICIPANT_STATUS = "Participant_Status.csv"
FILENAME_MOCA = "Montreal_Cognitive_Assessment__MoCA_.csv"
FILENAME_UPDRS1A = "MDS-UPDRS_Part_I.csv"
FILENAME_UPDRS1B = "MDS-UPDRS_Part_I_Patient_Questionnaire.csv"  # patient questionnaire
FILENAME_UPDRS2 = "MDS_UPDRS_Part_II__Patient_Questionnaire.csv"
FILENAME_UPDRS3 = "MDS-UPDRS_Part_III.csv"
FILENAME_UPDRS3_CLEAN = "MDS_UPDRS_Part_III_clean.csv"
FILENAME_UPDRS4 = "MDS-UPDRS_Part_IV__Motor_Complications.csv"
FILENAME_GDS = "Geriatric_Depression_Scale__Short_Version_.csv"
FILENAME_DOMSIDE = "PD_Features-Archived.csv"

# other file names
FILENAME_FMRI_INFO = "PPMI_rs-fMRI_Data-and-Methods/PPMI_fs-fMRI_dataReport_spreadsheet_FINAL_20191121.xlsx"
FILENAME_FMRI_INFO_ZIP = "PPMI_rs-fMRI_Data-and-Methods.zip"
FILENAME_FMRI_METADATA = "fMRI_info.csv"


# useful column names
COL_PAT_ID = "PATNO"
COL_STATUS = "COHORT_DEFINITION"
COL_VISIT_TYPE = "EVENT_ID"
COL_ENROLLMENT = 'ENROLL_STATUS'
COL_DATE_INFO = "INFODT"
COL_DATE_BIRTH = "BIRTHDT"
COL_DATE_PD = "PDDXDT"
COLS_DATE = [
    COL_DATE_INFO,
    "LAST_UPDATE",
    "ORIG_ENTRY",
    COL_DATE_BIRTH,
    COL_DATE_PD,
]
COL_PD_STATE = "PDSTATE"
COL_VISIT_TYPE = "EVENT_ID"
COL_PD_STATE = "PDSTATE"
COL_AGE = "AGE_AT_VISIT"
COL_SEX = "SEX"
COL_EDUCATION = "EDUCYRS"
COL_MOCA = "MCATOT"
COL_UPDRS1A = "NP1RTOT"
COL_UPDRS1B = "NP1PTOT"
COL_UPDRS1 = f"{COL_UPDRS1A}+{COL_UPDRS1B}"
COL_UPDRS2 = "NP2PTOT"
COL_UPDRS3 = "NP3TOT"
COL_UPDRS4 = "NP4TOT"

COL_FOLLOWUP = "is_followup"
COL_IMAGING_PROTOCOL = "Imaging Protocol" 
COL_IMAGE_DESC = "Description"

# codes for COHORT_DEFINITION field
STATUS_PD = "Parkinson's Disease"
STATUS_HC = "Healthy Control"

# codes for EVENT_ID field
REGEX_VISIT_FOLLOWUP = "^V((0[4-9])|(1[0-9])|(20))$"  # V04-V20

# codes for SEX field
SEX_FEMALE = 0
SEX_MALE = 1

# Acquisition parameters
FIELD_STRENGTH = "Field Strength=3.0"
MANUFACTURER = "SIEMENS"
IMAGE_DESC = [
    "ep2d_RESTING_STATE", 
    "ep2d_bold_rest",
]
TR_VAL = "TR=2400.0"
STATUS_MED = "ON"

# codes for SEX field
SEX_FEMALE = 0
SEX_MALE = 1

# column names/values obtained from searching the Image and Data Archive (IDA)
# are different from those used in the PPMI csv study files
# so they need to be remapped
IDA_COLNAME_MAP = {
    "Subject ID": COL_PAT_ID,
    "Visit": COL_VISIT_TYPE,
    "Study Date": COL_DATE_INFO,
    "Research Group": COL_STATUS
}
IDA_VISIT_MAP = {
    "Baseline": "BL",
    "Screening": "SC",
    "Month 6": "V02",
    "Month 12": "V04",
    "Month 24": "V06",
    "Month 36": "V08",
    "Month 48": "V10",
    "Symptomatic Therapy": "ST",
    "Unscheduled Visit 01": "U01",
    "Unscheduled Visit 02": "U02",
    "Premature Withdrawal": "PW",
}
IDA_STATUS_MAP = {
    "PD": "Parkinson's Disease",
    "Control":"Healthy Control",
    "SWEDD": "SWEDD",
    "Prodromal": "Prodromal",
    "GenCohortPD":"Parkinson's Disease",
    "GenCohortUnaff":"Prodromal"
}

ORIG_SUBLIST = [3107,3108,3113,3116,3123,3124,3125,3126,3127, 3128, 3130, 3134, 3327, 3332, 3352, 3354, 3359, 3360, 3364,
 3365, 3366, 3367, 3371, 3372,3373, 3375, 3378, 3380, 3383, 3385, 3386, 3387, 3392, 3552,3557, 3567, 3574, 3575, 3577, 3586, 3587, 3588,
 3589, 3591, 3592, 3593, 3760, 3800, 3808, 3814, 3815, 3818, 3819,3822, 3823, 3825, 3826, 3828, 3829, 3830, 3831, 3832, 3834, 3838, 3870,
 4020, 4024, 4026, 4030, 4034, 4035, 40366, 4038, 40533, 50485, 50901, 51632, 51731, 52678, 53060, 55395, 70463]

ORIG_SESLIST = ['5/15/2013', '4/24/2013', '7/17/2013', '11/14/2012', '6/19/2013', '7/17/2013', '7/10/2013', '9/18/2013', '1/25/2017',
            '9/19/2013', '11/16/2012', '4/22/2013', '11/29/2012','4/23/2013', '3/13/2013', '3/28/2013', '9/12/2013', '7/31/2013',
            '6/25/2013', '10/09/2013', '9/11/2013', '8/15/2013', '2/01/2013', '2/27/2013', '8/01/2013', '7/05/2013', '7/03/2013',
            '8/21/2013', '10/10/2012', '12/06/2012', '12/20/2012', '1/10/2013', '4/29/2013', '1/07/2013', '2/24/2015', '6/29/2015',
            '10/30/2013', '10/31/2012', '12/07/2015', '8/18/2014','9/17/2014', '9/16/2016', '1/25/2013', '3/08/2013', '4/15/2013',
            '3/27/2013', '1/23/2013', '3/19/2013', '11/19/2013', '12/10/2013', '11/03/2015', '4/08/2014', '4/09/2013', '5/28/2013',
            '6/04/2013', '7/30/2013', '8/20/2013', '9/17/2013', '10/01/2013', '11/26/2013', '10/29/2013', '12/03/2013', '1/14/2014',
            '4/01/2014', '12/17/2012', '2/16/2016', '8/02/2016', '8/30/2016','1/03/2013', '4/02/2013', '3/13/2013', '9/24/2014', 
            '4/01/2013', '6/21/2017', '3/09/2016', '12/17/2014', '11/18/2015', '9/29/2015', '11/18/2015', '11/09/2016', '1/05/2017',
             '8/31/2017']

Y1_SUBLIST = [3107,3108,3113,3116,3125,3130,3354,3359,3364,3366,3367,3371,3372,3373,3378,3383,3385,3386,3387,3552,3567,3577,3586,3587,3588,3589,3592,3800,3808,3814,3815,3818,3819,3822,3823,3826,3828,3829,3830,3831,3832,3834,3838,3870,4020,4026,4035,40366,50485,50901,51632,51731,52678]

Y2_SUBLIST = [3107,3108,3116,3123,3124,3126,3130,3332,3354,3366,3367,3371,3372,3378,3380,3383,3386,3552,3557,3567,3574,3575,3577,3586,3592,3800,3808,3814,3815,3819,3822,3823,3825,3826,3828,3829,3830,3831,3832,3834,4020,4024,40366,50901,52678]

Y4_SUBLIST = [3107,3108,3123,3124,3125,3126,3130,3134,3332,3354,3365,3367,3383,3575,3592,3593,3800,3808,3814,3819,3822,3823,3825,3826,3830,3831,3832,3834,3838,3870,4030,4034,4038]