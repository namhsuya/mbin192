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

chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        altchi1=dict(
            VAL=['N', 'CA', 'CB', 'CG2'],
        ),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        altchi2=dict(
            ASP=['CA', 'CB', 'CG', 'OD2'],
            LEU=['CA', 'CB', 'CG', 'CD2'],
            PHE=['CA', 'CB', 'CG', 'CD2'],
            TYR=['CA', 'CB', 'CG', 'CD2'],
        ),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
    )


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


def getChi1Chi2(id):

    chi1_chi2_out = []

    for chain in get_structure(id):
        for residue in chain:
            # Skip heteroatoms
            if residue.id[0] != " ": continue
            res_name = residue.resname
            if res_name in ("ALA", "GLY"): continue

            atoms = residue.get_list()
            atom_data = {}
            for atom in atoms:
                atom_data[atom.fullname.replace(" ", "")] = atom.get_vector()

            if res_name in chi_atoms["chi1"].keys():
                chi1_vec = [atom_data[a] for a in chi_atoms["chi1"][res_name]]
            chi1 = pdb.calc_dihedral(*chi1_vec)

            if res_name in chi_atoms["chi2"].keys():
                chi2_vec = [atom_data[a]  for a in chi_atoms["chi2"][res_name]]
            chi2 = pdb.calc_dihedral(*chi2_vec)
        
            chi1_chi2_out.append([residue.id[1], res_name, degrees(chi1), degrees(chi2)])

    chi1_chi2_out = pd.DataFrame(chi1_chi2_out, columns=["Position", "Residue", "Chi1", "Chi2"])
    return chi1_chi2_out

def getChi1Chi2plot(id, df):
    
    sns.set_style('white')
    sns.set_context("paper", font_scale = 2)
    sns.set_style("whitegrid")

    data = df
    data.drop("Residue", axis=1, inplace=True)
    df_melted = data.melt("Position",var_name="SideChainTorsion", value_name="Angle")
    g=sns.relplot(data=df_melted, x="Position", y="Angle", hue="SideChainTorsion", kind="line", height=8, aspect=1.2)
    g.fig.subplots_adjust(top=.95)
    plt.title("Chi1/Chi2 lineplot for PDB:"+id.lower())
    plt.tight_layout()
    plt.savefig('static/'+str(id.lower())+'_cc.png', bbox_inches='tight')

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
    
@app.get("/chi")
def post_cn(request: Request):
    result = "Enter PDB ID"
    return templates.TemplateResponse("cc.html", context={
        'request': request, 'result': result
    })

@app.post("/chi")
async def post_cn(request: Request, pdb_id: str = Form(...)):
    result = getChi1Chi2(pdb_id)
    getChi1Chi2plot(pdb_id, result)
    result = result.to_html()
    return templates.TemplateResponse("cc.html", context={
        'request': request, 'result': result, 'pdb_id': pdb_id
    })
