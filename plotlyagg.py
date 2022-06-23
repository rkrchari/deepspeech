import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime as dt

app = dash.Dash()

df = pd.read_csv('ProductionData.csv')

machine_options = df['machine'].unique()

app.layout = html.Div([
                html.Div([
                                dcc.Graph(id='graph-one')], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                                dcc.DatePickerRange(
                                                                id='date-picker-range',
                                                                min_date_allowed=dt(2016, 1, 1),
                                                                max_date_allowed=dt(2020, 12, 31),
                                                                start_date=dt(2016, 1, 1),
                                                                end_date=dt(2016, 2, 1),
                                                                initial_visible_month=dt(2016, 1, 1)
                                                                ),
                                dcc.Dropdown(
                                                                id='machine-choice',
                                                                options=[{'label': i, 'value': i} for i in machine_options],
                                                                value='',
                                                                searchable=True
                                                                ),
                                dcc.Markdown('''
                                                        Select Machine and Date range to view cost per piece.

                                                        ''')
                                ], style={'width': '48%', 'display': 'inline-block'})
                                ])


@app.callback(Output('graph-one', 'figure'),
              [Input('date-picker-range', 'start_date'),
              Input('date-picker-range', 'end_date'),
              Input('machine-choice', 'value')])
def update_output(start_date, end_date, machine_choice):
    df = pd.read_csv('ProductionData.csv')
    df['prod_date'] = pd.to_datetime(df['prod_date'], format='%Y-%m-%d')
    df = df[(df['prod_date'] > start_date) & (df['prod_date'] < end_date)]

    cost_per_piece_avg = df.groupby(['machine', df.prod_date.dt.year])['cost_per_piece'].transform('mean')
    
    figure = {
        'data': [
            go.Bar(
                x=df[df['machine'] == machine_choice]['prod_date'],
                y=df[df['machine'] == machine_choice]['cost_per_piece'],
                name='Cost Per Piece',
                marker={

                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}},
                hoverinfo='y',

                ),
            go.Scatter(
                x=df[df['machine'] == machine_choice]['prod_date'],
                y=cost_per_piece_avg ,
                name='Cost Per Piece Average',
                marker={
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}},
                hoverinfo='y',
                )
        ],
        'layout': go.Layout(
            title='Cost Per Piece',

            yaxis=dict(
                            title='Cost',
                            hoverformat='.2f'),
            xaxis={'title': machine_choice},
            hovermode='closest',

                )
    }

    return figure


if __name__ == '__main__':
    app.run_server()



