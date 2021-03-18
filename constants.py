
# size of 73
all_sitenames = [
    '三義', '三重', '中壢', '中山', '二林', '仁武', '冬山', '前金', '前鎮', '南投',
    '古亭', '善化', '嘉義', '土城', '埔里', '基隆', '士林', '大同', '大園', '大寮',
    '大里', '安南', '宜蘭', '小港', '屏東', '崙背', '左營', '平鎮', '彰化', '復興',
    '忠明', '恆春', '斗六', '新店', '新港', '新營', '新竹', '新莊', '朴子', '松山',
    '板橋', '林口', '林園', '桃園', '楠梓', '橋頭', '永和', '汐止', '沙鹿', '淡水',
    '湖口', '潮州', '竹山', '竹東', '線西', '美濃', '臺南', '臺東', '臺西', '花蓮',
    '苗栗', '菜寮', '萬華', '萬里', '西屯', '觀音', '豐原', '金門', '關山', '陽明',
    '頭份', '馬公', '馬祖', '鳳山', '麥寮', '龍潭', '富貴角'
    ]
sitenames = [
    '三義', '三重', '中壢', '中山', '二林', '仁武', '冬山', '前金', '前鎮', '南投', 
    '古亭', '善化', '嘉義', '土城', '埔里', '基隆', '士林', '大同', '大園', '大寮', 
    '大里', '安南', '宜蘭', '小港', '屏東', '崙背', '左營', '平鎮', '彰化', '復興', 
    '忠明', '恆春', '斗六', '新店', '新港', '新營', '新竹', '新莊', '朴子', '松山', 
    '板橋', '林口', '林園', '桃園', '楠梓', '橋頭', '永和', '汐止', '沙鹿', '淡水', 
    '湖口', '潮州', '竹山', '竹東', '線西', '美濃', '臺南', '臺東', '臺西', '花蓮', 
    '苗栗', '菜寮', '萬華', '萬里', '西屯', '觀音', '豐原', '關山', '陽明', '頭份', 
    '鳳山', '麥寮', '龍潭'
    ]

feature_cols = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
                'RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',
                'month', 'day', 'hour' 
                ]
field = [
    "sitename", 
    "best_loss",
    "best_rmse",
    "epoch",
    "timestamp"
]

#sample_sites = ["湖口", "林園", "南投", "士林", "埔里", "關山"]
#sample_sites = ["苗栗", "頭份", "新竹", "湖口", "觀音", "竹東"]
#sample_sites = ["林口", "桃園", "大園", "平鎮", "中壢", "龍潭"]
#sample_sites = ["前金", "左營", "前鎮", "復興", "橋頭", "楠梓", "仁武", "小港", "鳳山"]
#sample_sites = ["新莊", "板橋", "土城", "士林", "中山", "萬華", "新店", "永和", "淡水", "三重", "陽明", "大同", "松山", "古亭", "菜寮"]
sample_sites = ['陽明', '萬里', '淡水', '基隆', '士林', '林口', '三重', '中山', '菜寮', '大園', '汐止', '大同', '松山', '萬華', '觀音', '新莊', '古亭', '永和', '板橋']
