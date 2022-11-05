from fastapi import FastAPI, Request, Form
from pydantic import create_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import Bio.PDB as pdb


def get_structure(id):
    
    # Download
    id = id.upper()
    pdb.PDBList().retrieve_pdb_file(id, pdir=".", file_format="pdb")

    parser = pdb.PDBParser(PERMISSIVE=True, QUIET=True)
    structures = parser.get_structure(id, "pdb"+id.lower()+".ent")

    return structures[0]

def getCNdistance(id):

    Ci_Ni_plus_1_bond_distances = []

    for chain in get_structure(id):
        
        residues = chain.get_list()
        
        for i in range(0, len(residues)):
            
            if i == len(residues)-1:
                break

            atoms0 = residues[i].get_list()
            atoms1 = residues[i+1].get_list()

            if len(atoms0)>2 and len(atoms1)>2:
                Ci_Ni1_dist = np.linalg.norm(atoms0[2].coord - atoms1[0].coord)
                Ci_Ni_plus_1_bond_distances.append([residues[i].id[1], residues[i].resname, residues[i+1].resname, Ci_Ni1_dist])
    
    Ci_Ni_plus_1_bond_distances = pd.DataFrame(Ci_Ni_plus_1_bond_distances, columns=['Position', 'Residue-1', 'Residue-2', 'C--N bond distance'])
    Ci_Ni_plus_1_bond_distances.fillna(0)

    return Ci_Ni_plus_1_bond_distances[Ci_Ni_plus_1_bond_distances[Ci_Ni_plus_1_bond_distances.columns[3]]<2]

def getCNdistanceplot(id, df):

    sns.set_style('white')
    sns.set_context("paper", font_scale = 2)
    data = df
    data = data[data[data.columns[3]]<2]
    sns.set_style("whitegrid")

    plt.figure(figsize=(12,8))
    fig = sns.lineplot(data=data, x="Position", y=data.columns[3])
    fig.set_ylim(df[df.columns[3]].min()-0.1, df[df.columns[3]].max()+0.1)
    plt.title("C-N distance per residue plot for PDB:"+id.lower())
    plt.savefig('static/'+str(id.lower())+'_cn.png')


def resolve(f, s):
  if not f and s:
    return(0, s)
  elif f and not s:
    return(f, 0)
  else:
    return(f, s)



def degrees(rad_angle) :
    """Converts any angle in radians to degrees.

    If the input is None, the it returns None.
    For numerical input, the output is mapped to [-180,180]
    """
    if rad_angle is None :
        return None
    angle = rad_angle * 180 / math.pi
    while angle > 180 :
        angle = angle - 360
    while angle < -180 :
        angle = angle + 360
    return angle


def getPhiPsi(id):

    phi_psi_out = []

    for chain in get_structure(id):
        # residues = chain.get_list()
        residues = pdb.CaPPBuilder().build_peptides(chain)
        for poly_index, poly in enumerate(residues):
            phi_psi = poly.get_phi_psi_list()
            for res_index, residue in enumerate(poly):
                phi, psi = phi_psi[res_index]
                phi, psi = resolve(phi, psi)
                phi_psi_out.append([residue.id[1], residue.resname, degrees(phi), degrees(psi)])
    
    phi_psi_out = pd.DataFrame(phi_psi_out, columns=["Position", "Residue", "Phi", "Psi"])
    return phi_psi_out


def getPhiPsiplot(id, df):
    
    data = pd.DataFrame(df, columns=['Position', 'Residue', 'Phi', 'Psi'])
    data.drop("Residue", axis=1, inplace=True)
    sns.set_style("whitegrid")
    df_melted = data.melt("Position",var_name="Dihedrals", value_name="Angle")
    g = sns.relplot(data=df_melted, x="Position", y="Angle", hue="Dihedrals", kind="line", height=6, aspect=1.2)
    g.fig.subplots_adjust(top=.95)
    plt.title("Phi/Psi lineplot for PDB:"+id.lower())
    plt.tight_layout()
    plt.savefig('static/'+str(id.lower())+'_pp.png', bbox_inches='tight')



app = FastAPI()
templates = Jinja2Templates(directory='static/')
app.mount("/static", StaticFiles(directory="static"), name="static")

# Take PDB ID as input parameter
query_params = {
    "pdb_id": (str, "4xct"),
}

query_model = create_model("Query", **query_params)

@app.get("/")
# def read_form(params: query_model = Depends()):
def read_form(request: Request):
    return templates.TemplateResponse("home.html", context={'request':request})

@app.get("/cn")
def post_cn(request: Request):
    result = "Enter PDB ID"
    return templates.TemplateResponse("cn.html", context={
        'request': request, 'result': result
    })

@app.post("/cn")
async def post_cn(request: Request, pdb_id: str = Form(...)):
    result = getCNdistance(pdb_id)
    getCNdistanceplot(pdb_id, result)
    result = result.to_html()
    return templates.TemplateResponse("cn.html", context={
        'request': request, 'result': result, 'pdb_id': pdb_id
    })


@app.get("/pp")
def post_cn(request: Request):
    result = "Enter PDB ID"
    return templates.TemplateResponse("pp.html", context={
        'request': request, 'result': result
    })

@app.post("/pp")
async def post_cn(request: Request, pdb_id: str = Form(...)):
    result = getPhiPsi(pdb_id)
    getPhiPsiplot(pdb_id, result)
    result = result.to_html()
    return templates.TemplateResponse("pp.html", context={
        'request': request, 'result': result, 'pdb_id': pdb_id
    })
    

