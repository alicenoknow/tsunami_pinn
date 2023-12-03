import os
import datetime
from typing import TypedDict
import jinja2
from xhtml2pdf import pisa


class ReportContext(TypedDict):
    num: int
    date: datetime
    weight_r: float 
    weight_i: float 
    weight_r: float
    layers: int
    neurons: int
    epochs: int 
    lr: float
    total_loss: float
    residual_loss: float
    initial_loss: float
    boundary_loss: float
    img_loss: str
    img_loss_i: str
    img_loss_b: str
    img_loss_r: str
    mesh_name: str


def create_report(context: ReportContext, 
                  env_path: str, 
                  template_path: str, 
                  report_title: str) -> None:
    template_loader = jinja2.FileSystemLoader(env_path)
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(template_path)
    output_text = template.render(context)
    
    with open(os.path.join(report_title), "w+b") as out_pdf_file_handle:
        pisa.CreatePDF(src=output_text, dest=out_pdf_file_handle)