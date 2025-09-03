-- orbit_japa.j_user definition

CREATE TABLE orbit_japa.j_user
(

    `id` UUID,

    `created_at` DateTime,

    `is_deleted` UInt8,

    `user_idfr` UInt64,

    `user_identifier` UUID,

    `spiritual_institute_name` String,

    `spiritual_institute_idfr` UInt64,

    `spiritual_institute_identifier` UUID,

    `spiritual_institute_logo` String,

    `spiritual_temple_name` String,

    `spiritual_temple_id` String,

    `spiritual_temple_idfr` UInt64,

    `spiritual_temple_identifier` UUID,

    `spiritual_temple_area` String,

    `spiritual_temple_city` String,

    `spiritual_temple_state` String,

    `spiritual_temple_country` String,

    `spiritual_temple_zipcode` String,

    `spiritual_temple_logo` String,

    `spiritual_guru_name` String,

    `spiritual_guru_idfr` UInt64,

    `spiritual_guru_identifier` UUID,

    `spiritual_guru_profile_picture` String,

    `spiritual_guru_bio` String,

    `spiritual_guru_dob` Date,

    `spiritual_guru_gender` String,

    `spiritual_guru_is_as_mentor` UInt8,

    `max_malas_target_per_day` UInt64,

    `is_global_leader_board_enabled` UInt8,

    `donate_money_to_my_temple` Float64,

    `donate_money_to_my_guru` Float64,

    `japa_title` String,

    `city` String,

    `state` String,

    `country` String,

    `zipcode` String,

    `full_name` String,

    `profile_picture` String,

    `gender` String,

    `dob` Date
)
ENGINE = ReplacingMergeTree
ORDER BY user_identifier
SETTINGS index_granularity = 8192;

-- orbit_japa.j_user_chant definition

CREATE TABLE orbit_japa.j_user_chant
(

    `user_identifier` UUID,

    `start_time` DateTime,

    `end_time` DateTime,

    `counter` UInt64,

    `mala_count` Decimal(18,
 6),

    `session_duration_sec` UInt64,

    `start_time_angle` Float32,

    `end_time_angle` Float32,

    `created_at` DateTime,

    `japa_category` String
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(start_time)
ORDER BY (user_identifier,
 start_time)
SETTINGS index_granularity = 4096;


CREATE TABLE orbit_japa.j_user_chant_metrics_daily
(

    `user_identifier` UUID,

    `period_start` Date,

    `max_mala_count` Decimal(18,
 6),

    `min_mala_count` Decimal(18,
 6),

    `avg_mala_count` Float64,

    `max_duration_sec` UInt64,

    `min_duration_sec` UInt64,

    `avg_duration_sec` Float64,

    `max_counter` UInt64,

    `min_counter` UInt64,

    `avg_counter` Float64,

    `max_start_time_angle` Float32,

    `min_start_time_angle` Float32,

    `avg_start_time_angle` Float32,

    `max_end_time_angle` Float32,

    `min_end_time_angle` Float32,

    `avg_end_time_angle` Float32,

    `total_counter` UInt64,

    `total_mala_count` Decimal(18,
 6),

    `total_duration_sec` UInt64,

    `total_session_count` UInt64
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(period_start)
ORDER BY (user_identifier,
 period_start)
SETTINGS index_granularity = 8192;

