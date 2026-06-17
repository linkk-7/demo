
DROP TABLE IF EXISTS `monitor`;


CREATE TABLE `monitor` (
    id INTEGER PRIMARY KEY, 
    param_id INTEGER, 
    channel_id1 INTEGER,
    channel_id2 INTEGER,
    sp_id1 INTEGER,
    sp_id2 INTEGER,
    fps INTEGER,
    `time` INTEGER, 
    last_time INTEGER,
    work_status INTEGER,

    x_min_left INTEGER, 
    x_max_left INTEGER, 
    y_min_left INTEGER, 
    y_max_left INTEGER, 
    x_min_right INTEGER, 
    x_max_right INTEGER, 
    y_min_right INTEGER, 
    y_max_right INTEGER,

    normal_x REAL DEFAULT -0.05435280672552285,
    normal_y REAL DEFAULT -0.955125522876231,
    normal_z REAL DEFAULT -0.2911717842639709,

    use_reprojection_error_filter BOOLEAN DEFAULT 0,
    use_NN_distance_filter BOOLEAN DEFAULT 0,
    error_threshold REAL DEFAULT 1000,
    distance_threshold REAL DEFAULT 3.0,
    use_dbscan BOOLEAN DEFAULT 1,
    dbscan_eps DEFAULT 75,
    ransac_reproj_threshold REAL DEFAULT 17,
    max_xy_distance REAL DEFAULT 15,
    calibration_id INTEGER DEFAULT 0,
    range_filter_func TEXT DEFAULT "left_y > 233 and 0.65399*(left_x - 1459.07) - left_y < 0 and (1.47938*left_x+left_y)>1872.007 and left_y > 520 and (-left_x*1.0588+left_y+1270) > 0 and (left_y + 1.362 * left_x > 2122.10)" -- 过滤函数字符串
    );
