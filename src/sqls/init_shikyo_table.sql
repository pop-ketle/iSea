CREATE TABLE IF NOT EXISTS shikyo (
    place text,
    date text,
    fishing_type text,
    num_of_ship int,
    species text,
    catch float,
    high_price float,
    mean_price float,
    low_price float,
    primary key(place, date, fishing_type, species)
    )