"""
Constantes graphiques et helpers Plotly partagés (couleurs, libellés FR, bandes IQR,
mise en page). Les *fabriques de figures* renvoient une go.Figure : on sépare le
calcul de l'affichage.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, hex_to_rgb

# palette 
COLOR_REAL = "#1a365d"      
COLOR_PROJ = "#22a085"      
COLOR_NOISY = "#c0392b"     
COLOR_WPA_DAY = "#e67e22"   
COLOR_VFAST = "#9b59b6"     
COLOR_SVD = "royalblue"
COLOR_NMF = "crimson"

#  mécanismes additifs (partagé NB2 / NB4) 
COLOR_LAPLACE = "#c0392b"    # mécanisme de Laplace (rouge)
COLOR_GAUSSIAN = "#2471a3"   # mécanisme gaussien (bleu)
MECH_COLOR = {"laplace": COLOR_LAPLACE, "gaussian": COLOR_GAUSSIAN}
# convention de trait : plein = budget total ε réparti sur T, pointillé = budget ε chaque jour
MODE_DASH = {"sequential": "solid", "per_day": "dash"}
MODE_LABEL = {"sequential": "eps/T", "per_day": "eps/jour"}

#  palettes descriptives NB1 
PALETTE_PROFILS = ["#8ecae6", "#219ebc", "#126782", "#023047", "#4361ee"]   # profils individuels (froid)
PALETTE_AGG = ["#ffb703", "#fb8500", "#e07a5f", "#d62828", "#9d4edd"]        # jours de l'agrégat (chaud)
SCALE_SAISON = "Turbo"   # échelle continue mois / saison (thermosensibilité, etc.)
SCALE_CORR = "RdBu"      # échelle divergente centrée sur 0 (matrices de corrélation)

FR_DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
FR_MONTHS = ["", "janvier", "février", "mars", "avril", "mai", "juin",
             "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


def fr_date(d):
    """Datetime → 'Lundi 3 octobre 2022'."""
    return f"{FR_DAYS[d.weekday()]} {d.day} {FR_MONTHS[d.month]} {d.year}"


def rgba(color, alpha=0.15):
    """Couleur hex '#rrggbb' → chaîne 'rgba(r, g, b, alpha)' translucide (remplissages)."""
    r, g, b = hex_to_rgb(color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def band(x, q75, q25, fillcolor, name=None):
    """Trace de bande (remplissage entre les deux courbes q25 et q75)."""
    x = np.asarray(x)
    return go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([q75, q25[::-1]]),
        fill="toself", fillcolor=fillcolor,
        line=dict(color="rgba(255,255,255,0)"),
        name=name, showlegend=name is not None, hoverinfo="skip",
    )


def legend_below(y=-0.16):
    """Légende horizontale sous le graphe (évite le chevauchement avec le titre)."""
    return dict(orientation="h", yanchor="top", y=y, xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.97)", bordercolor="#d5dbdb", borderwidth=1)


def apply_default_layout(fig, title="", height=540, width=1000, legend_pos="below"):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15, color=COLOR_REAL)),
        height=height, width=width, template="plotly_white", hovermode="x unified",
        legend=legend_below() if legend_pos == "below"
        else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=95, b=110 if legend_pos == "below" else 70, l=80, r=40),
    )
    return fig


#  figure descriptive : répartition 3D Power × ToU (barres pleines)
TOU_NAMES = {0: "Base", 1: "Heures Creuses", 2: "Tempo"}
_BLUE_SCALE = [[0.0, "#a8dadc"], [0.5, "#457b9d"], [1.0, "#1d3557"]]


def _box_mesh(xc, yc, dz, dx=0.5, dy=0.5, color="#457b9d", name=""):
    x0, y0 = xc - dx / 2, yc - dy / 2
    xs = [x0, x0+dx, x0+dx, x0, x0, x0+dx, x0+dx, x0]
    ys = [y0, y0, y0+dy, y0+dy, y0, y0, y0+dy, y0+dy]
    zs = [0, 0, 0, 0, dz, dz, dz, dz]
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=[0,0,4,4,0,0,1,1,2,2,3,3], j=[1,2,5,6,1,5,2,6,3,7,0,4], k=[2,3,6,7,5,4,6,5,7,6,4,7],
        color=color, opacity=1.0, flatshading=True, hoverinfo="text", text=name,
        lighting=dict(ambient=0.65, diffuse=0.9, specular=0.18, roughness=0.45, fresnel=0.1),
        lightposition=dict(x=120, y=200, z=320))


def _box_edges(xc, yc, dz, dx=0.5, dy=0.5):
    x0, y0 = xc - dx / 2, yc - dy / 2
    v = [(x0,y0,0),(x0+dx,y0,0),(x0+dx,y0+dy,0),(x0,y0+dy,0),
         (x0,y0,dz),(x0+dx,y0,dz),(x0+dx,y0+dy,dz),(x0,y0+dy,dz)]
    seq = [0,1,2,3,0,None, 4,5,6,7,4,None, 0,4,None, 1,5,None, 2,6,None, 3,7,None]
    return ([v[s][0] if s is not None else None for s in seq],
            [v[s][1] if s is not None else None for s in seq],
            [v[s][2] if s is not None else None for s in seq])


def plot_power_tou_3d(counts, total=None, power_cats=(6, 9, 12), tou_cats=(0, 1, 2),
                      tou_names=None, title=None, height=680, width=950):
    """
    Histogramme 3D à barres pleines de la répartition Power × ToU.

    `counts`     : crosstab DataFrame (index = Power, colonnes = ToU).
    `total`      : effectif total pour les % (défaut : somme de counts).
    Renvoie une go.Figure (à .show() dans le notebook).
    """
    power_cats, tou_cats = list(power_cats), list(tou_cats)
    tou_names = tou_names or TOU_NAMES
    counts = counts.reindex(index=power_cats, columns=tou_cats).fillna(0).astype(int)
    total = int(total) if total is not None else int(counts.values.sum())
    cmax = max(int(counts.values.max()), 1)

    fig = go.Figure()
    ex, ey, ez = [], [], []
    for xi, P in enumerate(power_cats):
        for yi, Tc in enumerate(tou_cats):
            c = int(counts.loc[P, Tc])
            if c == 0:
                continue
            pct = 100 * c / total
            color = sample_colorscale(_BLUE_SCALE, c / cmax)[0]
            fig.add_trace(_box_mesh(xi, yi, c, color=color,
                          name=f"Power {P} kVA · {tou_names[Tc]}<br>{c} foyers ({pct:.1f}%)"))
            bx, by, bz = _box_edges(xi, yi, c)
            ex += bx + [None]; ey += by + [None]; ez += bz + [None]
            fig.add_trace(go.Scatter3d(x=[xi], y=[yi], z=[c + 0.04 * cmax], mode="text",
                          text=[f"<b>{c}</b><br>{pct:.1f}%"],
                          textfont=dict(size=11, color="#1d3557", family="Arial"),
                          showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                  line=dict(color="rgba(20,40,70,0.5)", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                  marker=dict(size=0.1, color=[cmax / 2], colorscale=_BLUE_SCALE, cmin=0, cmax=cmax,
                              colorbar=dict(title="nb foyers", thickness=14, len=0.55, x=0.92)),
                  showlegend=False, hoverinfo="skip"))
    axis_style = dict(backgroundcolor="rgba(247,249,252,1)", gridcolor="rgba(180,190,205,0.5)",
                      zeroline=False, showspikes=False)
    fig.update_layout(
        title=dict(text=title or f"Répartition des {total:,} foyers par classe Power × ToU".replace(",", " "),
                   x=0.5, font=dict(size=18, color="#1d3557")),
        scene=dict(
            xaxis=dict(title="Power (kVA)", tickvals=list(range(len(power_cats))),
                       ticktext=[str(p) for p in power_cats], **axis_style),
            yaxis=dict(title="ToU", tickvals=list(range(len(tou_cats))),
                       ticktext=[tou_names[t] for t in tou_cats], **axis_style),
            zaxis=dict(title="Nombre de foyers", **axis_style),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.85),
            camera=dict(eye=dict(x=1.7, y=1.6, z=1.05))),
        height=height, width=width, paper_bgcolor="white",
        margin=dict(l=0, r=0, t=60, b=0), showlegend=False)
    return fig