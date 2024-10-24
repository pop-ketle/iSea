CREATE TABLE IF NOT EXISTS suion (
    date text,
    time text,
    place text,
    water_temperature float,
    primary key(date,time,place)
)