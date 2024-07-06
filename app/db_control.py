import psycopg2
from pandas import DataFrame
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, date, timedelta
from names import (
    OM_Temperature,
    OM_Cloud_cover,
    OM_Humidity,
    OM_Precipitation,
    default_rename_columns_map,
)
from config import turn_off_warnings
from dotenv import load_dotenv
import os

load_dotenv()

turn_off_warnings()

conn = psycopg2.connect(dbname=os.getenv("POSTGRES_DB"),
                        user=os.getenv("POSTGRES_USER"),
                        password=os.getenv("POSTGRES_PASSWORD"),
                        host=os.getenv('host'),
                        port=os.getenv('port'))


def response_handler(response, district, lat, target_id):
    response_list = []
    for response, d, l, i in zip(response, district, lat, target_id):
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": hourly_temperature_2m,
            "precipitation": hourly_precipitation,
            "relative_humidity_2m": hourly_relative_humidity_2m,
            "cloud_cover": hourly_cloud_cover,
        }
        df = pd.DataFrame(data=hourly_data)
        df["district"] = d
        df["id"] = i
        df["latitude"] = l

        df = df.set_index('date').dropna()
        df_1 = df[[OM_Temperature, OM_Cloud_cover, OM_Humidity, 'district', 'id', 'latitude']].resample(
            '3h').asfreq()
        df_2 = df[OM_Precipitation].resample('3h').sum()
        merged_df = pd.concat([df_1, df_2], axis=1).reset_index()
        response_list.append(merged_df)
    return pd.concat(response_list).dropna()


def open_meteo_request(
        district, lon: list, lat: list, target_id: list, start_date: date
):
    print('weather data request')
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    if start_date >= datetime.utcnow().date() - timedelta(5):
        start_date = datetime.utcnow().date() - timedelta(10)
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params_forecast = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "relative_humidity_2m",
            "cloud_cover",
        ],
        "start_date": str(datetime.utcnow().date() - timedelta(1)),
        "end_date": str(datetime.utcnow().date() + timedelta(14)),
        "models": "best_match",
        "current": "temperature_2m",
        "timezone": "auto",
    }
    url_archive = "https://archive-api.open-meteo.com/v1/archive"
    params_archive = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(datetime.utcnow().date()),
        "hourly": [
            "temperature_2m",
            "precipitation",
            "relative_humidity_2m",
            "cloud_cover",
        ],
        "models": "best_match",
        "timezone": "auto",
    }

    responses_forecast = openmeteo.weather_api(url_forecast, params=params_forecast)
    responses_archive = openmeteo.weather_api(url_archive, params=params_archive)

    print('data processing')
    forecast = response_handler(responses_forecast, district, lat, target_id)
    archive = response_handler(responses_archive, district, lat, target_id)

    forecast = forecast.drop(
        forecast.loc[forecast["date"].dt.date == max(archive["date"]).date()].index
    )
    df_to_db = forecast.merge(archive, how="outer")
    print('writing data to the database')
    insert_data_on_conflict(data=df_to_db)


def get_meteo_data(start_date: date):
    # todo: Возможно стоит сделать здесь контроль за тем какие регионы обновлять
    df = pd.read_parquet("Parquet/centroids/Volgograd_centroids.parquet")
    open_meteo_request(
        lat=df['district_latitude'], lon=df['district_longitude'], district=df['district_name'],
        target_id=df['district_id'], start_date=start_date
    )


def init_db(connect=conn):
    with connect.cursor() as cur:
        cur.execute("""CREATE SCHEMA IF NOT EXISTS Volgograd_weather;""")

        cur.execute(f"""CREATE TABLE IF NOT EXISTS Volgograd_weather.Weather_data (
        weather_id SERIAL PRIMARY KEY,
        district_id INTEGER NOT NULL,
        district_name TEXT NOT NULL,
        weather_datetime TIMESTAMP NOT NULL,
        {OM_Temperature} REAL NOT NULL,
        {OM_Humidity} REAL NOT NULL,
        {OM_Cloud_cover} REAL NOT NULL,
        {OM_Precipitation} REAL NOT NULL,
        district_latitude REAL NOT NULL,
        CONSTRAINT district_datetime_key UNIQUE (district_name, weather_datetime)
        );""")

        cur.execute("""CREATE TABLE IF NOT EXISTS Volgograd_weather.Weather_data (
        weather_id SERIAL PRIMARY KEY,
        district_id INTEGER NOT NULL,
        district_name TEXT NOT NULL,
        weather_datetime TIMESTAMP NOT NULL,
        temperature_2m REAL NOT NULL,
        relative_humidity_2m REAL NOT NULL,
        cloud_cover REAL NOT NULL,
        precipitation REAL NOT NULL,
        district_latitude REAL NOT NULL,
        CONSTRAINT district_datetime_key UNIQUE (district_name, weather_datetime)
        );""")

        cur.execute("""CREATE TABLE IF NOT EXISTS Volgograd_weather.Cronjob_data(
        id SERIAL NOT NULL PRIMARY KEY,
        cronjob_activate_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
        description TEXT,
        status TEXT NOT NULL,
        tenant_id INTEGER NOT NULL,
        active_cronjob TEXT NOT NULL
        );""")
        connect.commit()


def insert_data_on_conflict(data: DataFrame, connect=conn):
    with connect.cursor() as cur:
        try:
            query = f"""INSERT INTO Volgograd_weather.weather_data (
            weather_datetime, {OM_Temperature}, {OM_Humidity}, {OM_Cloud_cover}, {OM_Precipitation}, district_name, district_id, district_latitude)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (district_name, weather_datetime) 
            DO UPDATE
            SET {OM_Temperature} = EXCLUDED.{OM_Temperature}, 
            {OM_Humidity} = EXCLUDED.{OM_Humidity}, 
            {OM_Cloud_cover} = EXCLUDED.{OM_Cloud_cover}, 
            {OM_Precipitation} = EXCLUDED.{OM_Precipitation}"""

            data.rename(columns=default_rename_columns_map, inplace=True)

            data_for_db = list(zip(data['weather_datetime'],
                                   data[OM_Temperature],
                                   data[OM_Humidity],
                                   data[OM_Cloud_cover],
                                   data[OM_Precipitation],
                                   data['district_name'],
                                   data['district_id'],
                                   data['district_latitude']))
            cur.executemany(query, data_for_db)
            connect.commit()
        except Exception as e:
            print("Ошибка:", e)
            connect.rollback()


def init_test_select(connect=conn):
    query = """SELECT * FROM Volgograd_weather.weather_data"""
    return len(pd.read_sql_query(query, connect))


def test_db(connect=conn):
    with connect.cursor() as cur:
        try:
            cur.execute("SELECT EXISTS (SELECT 1 FROM Volgograd_weather.weather_data);")
            table_exists = cur.fetchone()[0]
            if not table_exists:
                print('The database is empty, sending a request for weather data')
                get_meteo_data(start_date=datetime.utcnow().date() - timedelta(365))
        except Exception as e:
            print('Схемы не существует', e)


def select_data_only_time_ranged(start_time: str, end_time: str, connect=conn):
    query = f"""SELECT * FROM Volgograd_weather.weather_data 
    WHERE weather_datetime BETWEEN '{start_time} 00:00:00' AND '{end_time} 23:00:00';"""
    return pd.read_sql_query(query, connect)


def select_max_min_datetime(aggregate: str, connect=conn):
    if aggregate not in ['MIN', 'MAX']:
        raise ValueError("Аргумент 'aggregate' должен быть 'MIN' или 'MAX'")
    query = f"""SELECT {aggregate}(weather_datetime) FROM Volgograd_weather.weather_data"""
    with connect.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()[0][0]


def select_weather_data_for_forecast(aggregate: str,
                                     weather_indicator: str,
                                     date_range_start: str,
                                     date_range_end: str,
                                     connect=conn):
    if aggregate not in ['MIN', 'MAX', 'AVG', 'SUM']:
        raise ValueError("Аргумент 'aggregate' должен быть 'MIN', 'MAX', 'AVG', 'SUM'")
    query = f"""SELECT district_id, district_name, {aggregate}({weather_indicator}) FROM Volgograd_weather.weather_data WHERE weather_datetime
    BETWEEN '{date_range_start} 00:00:00' AND '{date_range_end} 23:00:00'
    GROUP BY district_name, district_id"""
    return pd.read_sql_query(query, connect)


if __name__ == '__main__':
    get_meteo_data(start_date=datetime.utcnow().date() - timedelta(20))
