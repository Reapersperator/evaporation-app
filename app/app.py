import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import time
import json
import pandas as pd
import geopandas as gpd
from pandas import DataFrame
from datetime import datetime, timedelta
import plotly.graph_objects as go
from calculations import data_calculation, handle_data_for_HUNIT
import plotly.express as px
from time import process_time
from names import (
    Temperature,
    Humidity,
    Precipitation,
    Cloud_cover,
    OM_Temperature,
    OM_Cloud_cover,
    OM_Humidity,
    OM_Precipitation,
)

import os
from const import culture_parquet_dir
from dash import dcc, html
from db_control import (init_db,
                        select_data_only_time_ranged,
                        select_weather_data_for_forecast,
                        select_max_min_datetime,
                        test_db, )
import calendar
from config import turn_off_warnings

turn_off_warnings()

# todo: init create db
print('Database initialization')
init_db()
print('Database check')
test_db()


def get_date_range(date):
    year = date.year
    month = date.month
    day = date.day

    if 1 <= day <= 10:
        start_current = datetime(year, month, 1)
        end_current = datetime(year, month, 10)
        end_previous = datetime(year, month, 1) - timedelta(days=1)
        start_previous = end_previous.replace(day=21)
    elif 11 <= day <= 20:
        start_current = datetime(year, month, 11)
        end_current = datetime(year, month, 20)
        end_previous = datetime(year, month, 10)
        start_previous = datetime(year, month, 1)
    else:

        _, last_day = calendar.monthrange(year, month)
        start_current = datetime(year, month, 21)
        end_current = datetime(year, month, last_day)
        end_previous = datetime(year, month, 20)
        start_previous = datetime(year, month, 11)

    return (start_current.date(), end_current.date()), (start_previous.date(), end_previous.date())


def convert_data_to_dict(data: DataFrame) -> dict:
    district_data_dict = {}
    for district in data["district_name"].unique():
        temporary_df = data.loc[(data["district_name"] == district)].set_index("weather_datetime")
        district_data_dict.update(
            {
                district: {
                    OM_Temperature: temporary_df[OM_Temperature]
                    .to_list(),
                    OM_Precipitation: temporary_df[OM_Precipitation]
                    .resample("D")
                    .sum()
                    .to_list(),
                    OM_Humidity: temporary_df[OM_Humidity]
                    .resample("D")
                    .mean()
                    .to_list(),
                    OM_Cloud_cover: temporary_df[OM_Cloud_cover]
                    .between_time(start_time="06:00", end_time="18:00")
                    .to_list(),
                    "latitude": temporary_df["district_latitude"].iloc[0],
                    "day_number": list(
                        temporary_df[OM_Temperature]
                        .resample("D")
                        .asfreq()
                        .index.strftime("%j")
                        .astype(int)
                    ),
                }
            }
        )
    return district_data_dict


def soil_data_calculations(
        culture,
        sowing_date,
        model_start=datetime.utcnow().date() + timedelta(1),
        model_end=datetime.utcnow().date() + timedelta(10)
):
    data_dict = convert_data_to_dict(
        select_data_only_time_ranged(start_time=str(model_start), end_time=str(model_end)))
    df_soil = gpd.read_file("GEOJSON/soil/Volgograd_soil.geojson")
    epes_list = []
    leaf_index = pd.read_parquet(f"Culture_parquet/{culture}.parquet")

    leaf_index_list = leaf_index.set_index(
        pd.date_range(sowing_date, sowing_date + timedelta(days=len(leaf_index) - 1))
    )
    index_a = pd.DataFrame(index=pd.date_range(model_start, model_end))

    leaf_index_list = leaf_index_list.loc[
        (leaf_index_list.index <= max(index_a.index))
        & (leaf_index_list.index >= min(index_a.index))
        ]

    # В случае когда значения ушли за пределы листового индекса, вернет нули
    leaf_index_list = pd.concat([index_a, leaf_index_list], axis=1).fillna(0)
    roots = leaf_index_list["Root"].reset_index(drop=True).to_list()
    leaf_index_list = leaf_index_list["LAI"].reset_index(drop=True).to_list()

    hunit_data = convert_data_to_dict(
        select_data_only_time_ranged(start_time=sowing_date, end_time=str(model_end)))
    hunit_dict = handle_data_for_HUNIT(
        hunit_data, hunit_len=len(leaf_index_list)
    )

    for district, albedo, U, alpha, NV, VZ in zip(
            df_soil["district"],
            df_soil["albedo"],
            df_soil["U"],
            df_soil["alpha"],
            df_soil["НВ, мм (слой 20 см)"],
            df_soil["ВЗ, мм"],
    ):
        epes_list.append(
            data_calculation(
                temperature_series=data_dict[district][OM_Temperature],
                cloud_cover_series=data_dict[district][OM_Cloud_cover],
                humidity_series=data_dict[district][OM_Humidity],
                precipitation_series=data_dict[district][OM_Precipitation],
                lat=data_dict[district]["latitude"],
                U=U,
                Alpha=alpha,
                soil_albedo=albedo,
                leaf_index_a=leaf_index_list,
                day_number=data_dict[district]["day_number"],
                NV=NV,
                VZ=VZ,
                hunit=hunit_dict[district],
                roots=roots,
            )
        )
    forecast_list = []
    archive_list = []
    for _ in epes_list:
        forecast_list.append(pd.DataFrame(_[:10]).sum())
        archive_list.append(pd.DataFrame(_[10:]).sum())

    data_for_forecast_figure = pd.concat(
        [df_soil.reset_index(drop=True), pd.concat(forecast_list).reset_index(drop=True)],
        axis=1,
    ).rename(columns={0: "EP+ES"})
    data_for_archive_figure = pd.concat(
        [df_soil.reset_index(drop=True), pd.concat(archive_list).reset_index(drop=True)],
        axis=1,
    ).rename(columns={0: "EP+ES"})

    return {'forecast': data_for_forecast_figure, 'archive': data_for_archive_figure}


def create_evaporation_data_figure(map_data):
    st = time.process_time()

    map_path = "GEOJSON/soil/Volgograd_soil.geojson"

    with open(map_path, encoding="utf-8") as f:
        geojson = json.load(f)
    figure = px.choropleth_mapbox(
        map_data,
        geojson=geojson,
        locations="id",
        featureidkey="properties.id",
        color='EP+ES',
        mapbox_style="open-street-map",
        color_continuous_scale='amp',
        center={"lat": 49.5, "lon": 44.0},
        opacity=0.4,
        hover_data={'id': False, 'district': True, 'Тип почвы': True, 'Гранулометрический состав': True, 'EP+ES': True},
        labels={'district': "Район", 'EP+ES': "Эвапотранспирация"},
        zoom=6,
    )
    print(f"Время формирования и расчета фигуры: {process_time() - st}")
    return figure


def create_empty_figure():
    return go.Figure(
        layout={
            "dragmode": "pan",
            "plot_bgcolor": "white",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        }
    )


def create_weather_data_figure(
        aggregate_type: str,
        indicator: str,
        model_start=datetime.utcnow().date() + timedelta(1),
        model_end=datetime.utcnow().date() + timedelta(10),
):
    st = time.process_time()
    map_path = "GEOJSON/Volgograd_for_weather.geojson"
    map_data = select_weather_data_for_forecast(aggregate=aggregate_type,
                                                date_range_start=str(model_start),
                                                date_range_end=str(model_end),
                                                weather_indicator=indicator)
    color_schema_map = {OM_Temperature: 'thermal', OM_Precipitation: 'ice_r', OM_Humidity: 'viridis',
                        OM_Cloud_cover: 'Blues'}
    indicator_names_map = {OM_Temperature: Temperature, OM_Precipitation: Precipitation, OM_Humidity: Humidity,
                           OM_Cloud_cover: Cloud_cover}
    with open(map_path, encoding="utf-8") as f:
        geojson = json.load(f)
    figure = px.choropleth_mapbox(
        map_data,
        geojson=geojson,
        locations="district_id",
        featureidkey="properties.id",
        color=map_data[aggregate_type.lower()],
        mapbox_style="open-street-map",
        color_continuous_scale=color_schema_map[indicator],
        center={"lat": 49.5, "lon": 44.0},
        opacity=0.4,
        hover_data={aggregate_type.lower(): f":.0f", "district_id": False, "district_name": True},
        labels={aggregate_type.lower(): indicator_names_map[indicator], "district_name": 'Район'},
        zoom=6,
    )
    print(f'Время формирования фигуры: {process_time() - st}')
    return figure


# Application
app = dash.Dash("Декадный прогноз", external_stylesheets=[dbc.themes.BOOTSTRAP])

culture_parquet_files = os.listdir(culture_parquet_dir)
path_val_cultures = [
    f.replace(".parquet", "")
    for f in culture_parquet_files
    if os.path.isfile(os.path.join(f"{culture_parquet_dir}/", f))
]

dropdown_weather_indicator = dcc.Dropdown(
    id="selection_dropdown_weather_indicator",
    options=[
        {"label": "Осадки", "value": OM_Precipitation},
        {"label": "Температура", "value": OM_Temperature},
        {"label": "Влажность", "value": OM_Humidity},
        {"label": "Облачность", "value": OM_Cloud_cover},
    ],
    value=None,
    placeholder="Выбор показателя",
)

dropdown_weather_indicator_aggregate_function = dcc.Dropdown(
    id="selection_dropdown_weather_indicator_aggregate_function",
    options=[
        {"label": "Минимальное значение", "value": 'MIN'},
        {"label": "Максимальное значение", "value": 'MAX'},
        {"label": "Среднее значение", "value": 'AVG'},
        {"label": "Сумма", "value": 'SUM'},
    ],
    value=None,
    placeholder="Выбор функции",
)

date_picker_forecast_start = dcc.DatePickerSingle(
    id="date_picker_weather_data_start",
    first_day_of_week=1,
    min_date_allowed=select_max_min_datetime(aggregate='MIN') + timedelta(1),
    max_date_allowed=select_max_min_datetime(aggregate='MAX') - timedelta(1),
    date=datetime.utcnow().date() + timedelta(1),
    show_outside_days=False,
    display_format="DD-MM-YYYY",
)

date_picker_forecast_end = dcc.DatePickerSingle(
    id="date_picker_weather_data_end",
    first_day_of_week=1,
    min_date_allowed=select_max_min_datetime(aggregate='MIN') + timedelta(1),
    max_date_allowed=select_max_min_datetime(aggregate='MAX') - timedelta(1),
    date=datetime.utcnow().date() + timedelta(10),
    show_outside_days=False,
    display_format="DD-MM-YYYY",
)

dropdown_cultures = dcc.Dropdown(
    id="selection_dropdown_culture",
    options=[{"label": i, "value": i} for i in path_val_cultures],
    value=None,
    placeholder="Выбор культуры",
)

date_picker_sowing_date = dcc.DatePickerSingle(
    id="date_picker_for_sowing_date",
    first_day_of_week=1,
    min_date_allowed=datetime.utcnow() - timedelta(days=365),
    max_date_allowed=datetime.utcnow().date() - timedelta(21),
    placeholder="Дата сева",
    show_outside_days=False,
    date=datetime.utcnow().date() - timedelta(21),
    display_format="DD-MM-YYYY",
)

table_content_1 = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Погода"),
                    html.Div(dropdown_weather_indicator, style={"width": "50vh"}),
                    html.Div(dropdown_weather_indicator_aggregate_function, style={"width": "50vh"}),
                    dbc.Button(
                        "Показать прогноз",
                        id="submit_button_for_forecast",
                        n_clicks=0,
                        style={"color": "primary", "margin-top": 20, "margin-bot": 20})
                ]
            ),
            dbc.Col(
                [
                    html.Div('Начало'),
                    html.Div(date_picker_forecast_start),
                    html.Div('Конец'),
                    html.Div(date_picker_forecast_end)

                ]

            )
        ],
    ),
    dbc.Row(
        dcc.Loading(
            id="loading-1",
            children=dcc.Graph(
                id="graph", config={"scrollZoom": True}, style={"height": "80vh"}
            ),
        )
    ),
]

table_content_2 = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Культура"),
                    html.Div(dropdown_cultures, style={"width": 500}),
                    dbc.Button(
                        "Рассчитать",
                        id="submit-button",
                        n_clicks=0,
                        style={"color": "primary", "margin-top": 20, "margin-bot": 20},
                    ),
                ]
            ),
            dbc.Col([html.Div("Дата сева"), date_picker_sowing_date]),
        ]
    ),
    dbc.Row([dbc.Col([html.H4(id='forecast_soil_figure_description'),
                      dcc.Loading(
                          id="loading-2",
                          children=dcc.Graph(
                              id="graph_soil_forecast", style={"height": "80vh"}, config={"scrollZoom": True}
                          ),
                      )]),
             dbc.Col([html.H4(id='archive_soil_figure_description'),
                      dcc.Loading(
                          id="archive_soil_figure_loading",
                          children=dcc.Graph(
                              id="graph_soil_archive", style={"height": "80vh"},
                              config={"scrollZoom": True}
                          ),
                      )])
             ]

            ),
]

app.layout = dbc.Container(
    [
        dbc.Tabs(
            [
                dbc.Tab(table_content_1, label="Обзор данных"),
                dbc.Tab(table_content_2, label="Эвапотранспирация"),
            ]
        ),
    ],
    fluid=True,
)


# todo: Callbacks
@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="submit_button_for_forecast", component_property="n_clicks"),
    [
        State(component_id='selection_dropdown_weather_indicator', component_property='value'),
        State(component_id='selection_dropdown_weather_indicator_aggregate_function', component_property='value'),
        State(component_id='date_picker_weather_data_start', component_property='date'),
        State(component_id='date_picker_weather_data_end', component_property='date')
    ]
)
def change_figure(n, indicator, aggregate, start_date, end_date):
    if None in [indicator, aggregate, start_date, end_date]:
        return create_empty_figure()
    elif end_date < start_date:
        return create_empty_figure()
    else:
        return create_weather_data_figure(
            aggregate_type=aggregate,
            indicator=indicator,
            model_start=start_date,
            model_end=end_date
        )


@app.callback(
    [Output(component_id='date_picker_weather_data_start', component_property='max_date_allowed'),
     Output(component_id='date_picker_weather_data_start', component_property='min_date_allowed'),
     Output(component_id='date_picker_weather_data_end', component_property='max_date_allowed'),
     Output(component_id='date_picker_weather_data_end', component_property='min_date_allowed'),
     Output(component_id='date_picker_for_sowing_date', component_property='min_date_allowed')],
    [Input(component_id="date_picker_weather_data_start", component_property="date"),
     Input(component_id="date_picker_weather_data_end", component_property="date"),
     Input(component_id="date_picker_for_sowing_date", component_property="date")])
def change_figure(date_start, date_end, date):
    min_date = select_max_min_datetime(aggregate='MIN') + timedelta(1)
    max_date = select_max_min_datetime(aggregate='MAX') - timedelta(1)
    return max_date, min_date, max_date, min_date, min_date


@app.callback(
    [
        Output(component_id="forecast_soil_figure_description", component_property="children"),
        Output(component_id="archive_soil_figure_description", component_property="children"),
        Output(component_id="graph_soil_forecast", component_property="figure"),
        Output(component_id="graph_soil_archive", component_property="figure")
    ],
    [
        Input(component_id="submit-button", component_property="n_clicks")
    ],
    [
        State(component_id="selection_dropdown_culture", component_property="value"),
        State(component_id="date_picker_for_sowing_date", component_property="date"),
    ],
)
def change_soil_figure(n, culture, sowing_date):
    if None in [culture, sowing_date]:
        return (None,
                None,
                create_empty_figure(),
                create_empty_figure()
                )
    elif culture and sowing_date:
        date_range = get_date_range(datetime.utcnow().date())
        range_forecast = date_range[0]
        range_archive = date_range[1]

        print(f'{range_archive[0]} ({range_archive[0] - timedelta(1)}) - {range_forecast[1]}')

        data = soil_data_calculations(culture=culture,
                                      sowing_date=datetime.strptime(sowing_date, "%Y-%m-%d").date(),
                                      model_start=range_archive[0] - timedelta(1),
                                      model_end=range_forecast[1])
        return (
            f"Прогноз эвапотранспирации с {range_forecast[0]} по {range_forecast[1]}",
            f"Предыдущая декада эвапотранспирации с {range_archive[0]} по {range_archive[1]}",
            create_evaporation_data_figure(data['forecast']),
            create_evaporation_data_figure(data['archive'])
        )


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8081)
