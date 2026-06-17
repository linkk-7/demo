DROP TABLE IF EXISTS `camera`;

CREATE TABLE `camera` (
    id INTEGER PRIMARY KEY,
    obj_param_id INTEGER,
    param_id INTEGER,
    local_ip TEXT,
    focal_zoom REAL,
    longitude REAL,
    latitude REAL,
    user TEXT,
    `password` TEXT,
    x_min INTEGER,
    x_max INTEGER,
    y_min INTEGER,
    y_max INTEGER,
    work_status INTEGER,
    sensor_param_id INTEGER
);