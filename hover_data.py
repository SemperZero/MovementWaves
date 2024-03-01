import io
import base64
from dash import Dash, dcc, html, Input, Output, no_update, callback
from PIL import Image

class DisplayWithHover(Dash):
    def __init__(self, fig, cluster_hover, labels, hover_width, text_labels, hover_individual, callback_type):
        super().__init__(__name__)

        self.cluster_hover = cluster_hover
        self.labels = labels
        self.hover_individual = hover_individual
        self.hover_width = hover_width
        self.text_labels = text_labels
        #print(self.hover_width,"HOVER WIDTH")

        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )

        fig.update_layout(
            width=800,
            height=600,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=2, y=2, z=2)
            )
        )

        self.layout = html.Div(
            className="container",
            children=[
                dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
            ],
        )
        if callback_type == "scatter":
            self.callback(
                Output("graph-tooltip-5", "show"),
                Output("graph-tooltip-5", "bbox"),
                Output("graph-tooltip-5", "children"),
                Input("graph-5", "hoverData"),
            )(self.display_hover)
        elif callback_type == "heatmap":
            self.callback(
                Output("graph-tooltip-5", "show"),
                Output("graph-tooltip-5", "bbox"),
                Output("graph-tooltip-5", "children"),
                Input("graph-5", "hoverData"),
            )(self.display_hover_heatmap)

    def np_image_to_base64_old(self, im_matrix):
        im = Image.fromarray(im_matrix)
        buffer = io.BytesIO()
        im.save(buffer, format="png")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/png;base64, " + encoded_image
        return im_url
    
    def np_image_to_base64(self, im_matrix):
        encoded_image = base64.b64encode(im_matrix).decode()
        im_url = "data:image/png;base64, " + encoded_image
        return im_url

    def display_hover(self, hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"] 
        print(hover_data)
        num = hover_data["pointNumber"]#make sure this is in the order the values were passed
        print("num", hover_data["pointNumber"])

        #print(num)
        #TODO: don't think this is correct. pass the flag to be 100%. colision between cluster number and num? or will it just be empty?
        if num in self.hover_individual:#if we have individual reconstruction
            hover_img_data = self.hover_individual[num]
            hover_width = self.hover_width[num]
        else:
            hover_img_data = self.cluster_hover[self.labels[num]]
           # print(self.hover_width)
            hover_width = self.hover_width[self.labels[num]]
    #    print(self.labels)
       # print(num, self.text_labels[num])


        #print(text)
        
        if hover_img_data is not None:
            im_url = self.np_image_to_base64(hover_img_data)
            children = [
                html.Div([
                    html.P(f"Cluster NR {str(self.labels[num])} | {self.text_labels[num]}, i: {num}", style={'font-weight': 'bold'}),
                    html.Img(
                        src=im_url,
                        style={"width": f"{hover_width}px", "height": "250px", 'display': 'block', 'margin': '0 auto'},
                    )   
                ])
            ]
        else: 
            children = [
                html.Div([
                    html.P(f"Cluster NR {str(self.labels[num])} | P: {self.text_labels[num]}", style={'font-weight': 'bold'}),
                ])
            ]
            
        

        return True, bbox, children


