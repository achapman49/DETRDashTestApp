import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import requests
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
from itertools import cycle
import pandas as pd
import plotly.express as px

# Define the external stylesheet for better aesthetics
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Object Detection in Images"),
    dcc.Input(id='image-url', type='text', placeholder='Enter Image URL', style={'width': '70%'}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-image'),
    html.Div(id='output-chart')
])

# Global variables to store labels and scores
global_labels_scores = []

def detect_objects(image_url):
    global global_labels_scores
    
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    image_np = np.array(image)

    boxes = []
    labels = []
    scores = []
    areas = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        area = (box[2] - box[0]) * (box[3] - box[1])
        boxes.append(box)
        labels.append(model.config.id2label[label.item()])
        scores.append(round(score.item(), 2))
        areas.append(area)

    sorted_indices = np.argsort(areas)[::-1]

    boxes = [boxes[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    global_labels_scores = list(zip(labels, scores))

    color_palette = px.colors.qualitative.Safe
    color_cycle = cycle(color_palette)
    color_map = {label: next(color_cycle) for label in set(labels)}

    return image_np, boxes, labels, scores, color_map

def generate_image(image_np, boxes, labels, scores, color_map):
    fig = px.imshow(image_np)

    for box, label, score in zip(boxes, labels, scores):
        fig.add_trace(go.Scatter(
            x=[box[0], box[2], box[2], box[0], box[0]], 
            y=[box[1], box[1], box[3], box[3], box[1]],
            mode='lines',
            line=dict(color=color_map[label], width=2),
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0)',
            hoverinfo='text',
            text=f'{label}: {score}',
            showlegend=False
        ))

    legend_items = [go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=label,
        showlegend=True,
        name=label
    ) for label, color in color_map.items()]

    for item in legend_items:
        fig.add_trace(item)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        xaxis_title=None,
        yaxis_title=None,
        hovermode='closest',
        legend=dict(title="Object Types", traceorder="normal")
    )

    return fig

def plot_object_scores(global_labels_scores, color_map):
    df = pd.DataFrame(global_labels_scores, columns=['Label', 'Score'])
    df['Color'] = df['Label'].map(color_map)
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_sorted['UniqueLabel'] = [f"{label} {i+1}" for i, (label, _) in enumerate(zip(df_sorted['Label'], df_sorted.index))]

    fig = px.bar(
        df_sorted,
        y=df_sorted.index,
        x='Score',
        orientation='h',
        color='Label',
        text=df_sorted.apply(lambda row: f"{round(row['Score'] * 100, 1)}% - {get_score_tag(row['Score'])}", axis=1),
        color_discrete_map=color_map
    )

    fig.update_layout(
        xaxis_title='Score',
        yaxis_title='Object Types',
        xaxis=dict(tickformat=".0%", showgrid=False),
        yaxis=dict(tickmode='array', tickvals=df_sorted.index, ticktext=df_sorted['Label'] + ' ' + (df_sorted.groupby('Label').cumcount() + 1).astype(str), autorange='reversed'),
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        title="Object Detection Scores"
    )

    return fig

def get_score_tag(score):
    percentage = score * 100
    if percentage >= 100:
        return "I'm certain"
    elif percentage > 95:
        return "I'm fairly sure"
    else:
        return "I'm uncertain"

@app.callback(
    [Output('output-image', 'children'),
     Output('output-chart', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('image-url', 'value')]
)
def update_output(n_clicks, image_url):
    if n_clicks > 0 and image_url:
        image_np, boxes, labels, scores, color_map = detect_objects(image_url)
        image_fig = generate_image(image_np, boxes, labels, scores, color_map)
        score_fig = plot_object_scores(global_labels_scores, color_map)

        return [
            dcc.Graph(figure=image_fig),
            dcc.Graph(figure=score_fig)
        ]
    return [html.Div(), html.Div()]

if __name__ == '__main__':
    app.run_server(debug=True)
