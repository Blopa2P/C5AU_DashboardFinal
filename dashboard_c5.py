import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")
st.title("📈 Dashboard de Análisis de Ventas")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", parse_dates=["Date"])
    return df

df = load_data()

# Sidebar - Filtros generales
st.sidebar.header("Filtros")
branch = st.sidebar.multiselect("Sucursal (Branch)", options=df["Branch"].unique(), default=df["Branch"].unique())
product_lines = st.sidebar.multiselect("Línea de Producto", options=df["Product line"].unique(), default=df["Product line"].unique())
df_filtered = df[(df["Branch"].isin(branch)) & (df["Product line"].isin(product_lines))]

# -------------------------------
# 1. Evolución de las Ventas Totales
# -------------------------------
st.subheader("1️⃣ Evolución de las Ventas Totales por Sucursal con Medias Móviles")

# Agrupar por fecha y sucursal
ventas_branch_fecha = df_filtered.groupby(["Date", "Branch"])["Total"].sum().reset_index()

# Calcular medias móviles
ventas_branch_fecha = ventas_branch_fecha.sort_values(by=["Branch", "Date"])
ventas_branch_fecha["Media Móvil 7 días"] = ventas_branch_fecha.groupby("Branch")["Total"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
ventas_branch_fecha["Media Móvil 25 días"] = ventas_branch_fecha.groupby("Branch")["Total"].transform(lambda x: x.rolling(window=25, min_periods=1).mean())

# Crear gráfico interactivo con líneas separadas
fig1 = px.line(
    ventas_branch_fecha,
    x="Date",
    y="Total",
    color="Branch",
    line_dash_sequence=["solid"],
    title="Ventas Diarias por Sucursal con Media Móvil (7 y 25 días)",
    labels={"Total": "Ventas ($)", "Date": "Fecha"}
)

# Agregar líneas punteadas para media móvil de 7 y 25 días
for branch in ventas_branch_fecha["Branch"].unique():
    df_branch = ventas_branch_fecha[ventas_branch_fecha["Branch"] == branch]
    fig1.add_scatter(
        x=df_branch["Date"],
        y=df_branch["Media Móvil 7 días"],
        mode="lines",
        name=f"{branch} - MM 7 días",
        line=dict(dash="dot")
    )
    fig1.add_scatter(
        x=df_branch["Date"],
        y=df_branch["Media Móvil 25 días"],
        mode="lines",
        name=f"{branch} - MM 25 días",
        line=dict(dash="dash")
    )

fig1.update_layout(legend_title_text="Sucursal / Línea")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# 2. Ingresos por Línea de Producto - Análisis Multidimensional
# -------------------------------
st.subheader("2️⃣ Ingresos por Línea de Producto - Análisis Multidimensional")

# Filtro adicional: Tipo de cliente para el gráfico de dispersión
st.markdown("**🎯 Filtrar gráfico de dispersión por tipo de cliente:**")
selected_customer_type = st.multiselect(
    "Selecciona tipo(s) de cliente",
    options=df_filtered["Customer type"].unique(),
    default=df_filtered["Customer type"].unique()
)

# Filtrar datos para el gráfico de dispersión según selección
scatter_data = df_filtered[df_filtered["Customer type"].isin(selected_customer_type)]

# 2.1 Gráfico de Barras Apiladas
ingresos_branch_product = df_filtered.groupby(["Product line", "Branch"])["Total"].sum().reset_index()
fig2_1 = px.bar(
    ingresos_branch_product,
    x="Total",
    y="Product line",
    color="Branch",
    orientation="h",
    barmode="stack",
    title="Ingresos Totales por Línea de Producto y Sucursal",
    labels={"Total": "Ingresos ($)", "Product line": "Línea de Producto", "Branch": "Sucursal"}
)

# 2.2 Gráfico de Dispersión filtrado
fig2_2 = px.scatter(
    scatter_data,
    x="Rating",
    y="Total",
    size="Quantity",
    color="Gender",
    hover_data=["Product line", "Branch"],
    title="Ingreso vs. Calificación (filtrado por tipo de cliente)",
    labels={"Total": "Ingreso por Venta", "Rating": "Calificación", "Quantity": "Cantidad", "Gender": "Género"},
    size_max=30
)

# 2.3 Gráfico de Barras Apiladas: Ingresos por Línea de Producto y Género
ingresos_genero = df_filtered.groupby(["Product line", "Gender"])["Total"].sum().reset_index()
fig2_3 = px.bar(
    ingresos_genero,
    x="Product line",
    y="Total",
    color="Gender",
    barmode="stack",
    title="Ingresos por Línea de Producto Segmentado por Género",
    labels={"Total": "Ingresos ($)", "Product line": "Línea de Producto", "Gender": "Género"}
)
st.plotly_chart(fig2_2, use_container_width=True)
# Mostrar en 3 columnas
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig2_1, use_container_width=True)
with col2:
    st.plotly_chart(fig2_3, use_container_width=True)



# -------------------------------
# 3. Distribución de la Calificación de Clientes
# -------------------------------
st.subheader("3️⃣ Distribución de la Calificación de Clientes")
'''
from scipy.stats import shapiro, skew, kurtosis
import numpy as np

# Filtro aplicado
df_rating = df_filtered.copy()
ratings = df_rating["Rating"]

# Estadísticas
shapiro_test = shapiro(ratings)
is_normal = shapiro_test.pvalue > 0.05
media = np.mean(ratings)
mediana = np.median(ratings)

if is_normal:
    asimetria_val = skew(ratings)
    curtosis_val = kurtosis(ratings)
    asimetria_str = f"**{asimetria_val:.2f}** {'(positiva)' if asimetria_val > 0 else '(negativa)' if asimetria_val < 0 else '(simétrica)'}"
    curtosis_str = f"**{curtosis_val:.2f}** {'(leptocúrtica)' if curtosis_val > 0 else '(platicúrtica)' if curtosis_val < 0 else '(mesocúrtica)'}"
else:
    asimetria_str = "No aplica"
    curtosis_str = "No aplica"
    # Mostrar advertencia
    st.warning("La distribución no es normal según el test de Shapiro-Wilk.")


# Texto de análisis
texto = f"""
**Análisis de la Calificación de Clientes**
- ¿Distribución normal? **{"Sí" if is_normal else "No"}** (p = {shapiro_test.pvalue:.3f})
- Media: **{media:.2f}**
- Mediana: **{mediana:.2f}**
- Coef. de asimetría: {asimetria_str}
- Coef. de curtosis: {curtosis_str}
"""

# Generación de gráfico y análisis en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")
    sns.histplot(data=ratings, kde=True, bins=20, color="white", ax=ax)
    ax.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
    ax.axvline(mediana, color='blue', linestyle='--', label=f'Mediana: {mediana:.2f}')
    ax.legend()
    ax.grid(True)
    ax.set_facecolor('none')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    st.pyplot(fig)


with col2:
    st.markdown(texto)
'''
# -------------------------------
# 4. Comparación del Gasto por Tipo de Cliente
# -------------------------------
st.subheader("4️⃣ Comparación del Gasto por Tipo de Cliente")

# Aplicar filtro de sucursal
df_box = df_filtered.copy()

# Calcular estadísticas descriptivas por tipo de cliente
stats = df_box.groupby("Customer type")["Total"].describe()[["mean", "50%", "25%", "75%"]].reset_index()
stats.columns = ["Tipo de Cliente", "Media", "Mediana", "Q1", "Q3"]

# Gráfico de caja
fig4 = px.box(
    df_box,
    x="Customer type",
    y="Total",
    color="Customer type",
    title="Distribución del Gasto por Tipo de Cliente",
    labels={"Total": "Gasto Total ($)", "Customer type": "Tipo de Cliente"},
    color_discrete_sequence=px.colors.qualitative.Set2  # Colores consistentes
)
fig4.update_layout(
    showlegend=False,  # Ocultar leyenda si los colores son obvios
    title_x=0.5,  # Centrar título
    plot_bgcolor="rgba(0,0,0,0)",  # Fondo transparente
    paper_bgcolor="rgba(0,0,0,0)"
)

# Mostrar gráfico de caja
st.plotly_chart(fig4, use_container_width=True)

# Mostrar tabla de estadísticas
st.markdown("**Estadísticas Descriptivas**")
st.dataframe(
    stats.style.format({"Media": "{:.2f}", "Mediana": "{:.2f}", "Q1": "{:.2f}", "Q3": "{:.2f}"}),
    use_container_width=True
)

# Nuevo gráfico: Gasto promedio por tipo de cliente y sucursal
st.markdown("**Gasto Promedio por Tipo de Cliente y Sucursal**")
gasto_promedio = df_box.groupby(["Customer type", "Branch"])["Total"].mean().reset_index()
fig4_2 = px.bar(
    gasto_promedio,
    x="Customer type",
    y="Total",
    color="Branch",
    barmode="group",
    title="Gasto Promedio por Tipo de Cliente y Sucursal",
    labels={"Total": "Gasto Promedio ($)", "Customer type": "Tipo de Cliente", "Branch": "Sucursal"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig4_2.update_layout(
    title_x=0.5,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="Sucursal"
)

# Mostrar gráfico de barras agrupadas
st.plotly_chart(fig4_2, use_container_width=True)

# -------------------------------
# 5. Relación entre Costo y Ganancia Bruta
# -------------------------------
import numpy as np
import scipy.stats as stats

st.subheader("5️⃣ Relación entre Costo y Ganancia Bruta")


scatter_data = df_filtered.copy()

# Calcular correlación
corr_coef, p_value = stats.pearsonr(scatter_data["cogs"], scatter_data["gross income"])
corr_text = f"""
**Análisis de Correlación**
- Coeficiente de correlación (Pearson): **{corr_coef:.2f}**
- Valor p: **{p_value:.3f}**
- Interpretación: {'Fuerte correlación positiva' if corr_coef > 0.7 else 'Correlación positiva moderada' if corr_coef > 0.3 else 'Correlación débil o nula'} entre costo y ganancia bruta.
- **Nota:** El coeficiente de correlación de Pearson mide la fuerza y dirección de la relación lineal entre dos variables.
"""
 # Mostrar advertencia sobre correlación perfecta
st.warning(
    f"Correlación perfecta debido al porcentaje de margen bruto de media **{scatter_data['gross margin percentage'].mean():.2f}%** "
    f"y **{scatter_data['gross margin percentage'].std():.2f}%** de desviación estándar."
    )
# Gráfico de dispersión con línea de regresión
fig5 = px.scatter(
    scatter_data,
    x="cogs",
    y="gross income",
    color="Product line",
    hover_data=["Quantity", "Customer type", "Branch"],
    title="Relación entre Costo y Ganancia Bruta",
    labels={"cogs": "Costo de Bienes Vendidos ($)", "gross income": "Ingreso Bruto ($)"},
    color_discrete_sequence=px.colors.qualitative.Set2
)

# Añadir línea de regresión
if len(scatter_data) > 1:  # Evitar error si hay pocos datos
    z = np.polyfit(scatter_data["cogs"], scatter_data["gross income"], 1)
    p = np.poly1d(z)
    fig5.add_scatter(
        x=scatter_data["cogs"],
        y=p(scatter_data["cogs"]),
        mode="lines",
        name="Línea de Regresión",
        line=dict(color="red", dash="dash")
    )

fig5.update_layout(
    title_x=0.5,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="Línea de Producto"
)

# Mostrar gráfico y resumen en dos columnas
col1, col2 = st.columns([3, 1])
with col1:
    st.plotly_chart(fig5, use_container_width=True)
with col2:
    st.markdown(corr_text)

# -------------------------------
# 6. Métodos de Pago Preferidos
# -------------------------------
st.subheader("6️⃣ Métodos de Pago Preferidos")
metodos_pago = df_filtered["Payment"].value_counts().reset_index()
metodos_pago.columns = ["Método de Pago", "Cantidad"]
fig6 = px.pie(metodos_pago, names="Método de Pago", values="Cantidad", title="Preferencia de Métodos de Pago", hole=0.4)
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------
# 7. Análisis de Correlación Numérica
# -------------------------------
import plotly.express as px
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("7️⃣ Análisis de Correlación Numérica")

# Texto introductorio
st.markdown("""
**Explora relaciones entre tres variables numéricas en un gráfico 3D interactivo.**
- **Cómo usarlo**: Selecciona tres variables (e.g., Total, Costo, Calificación) y filtra por sucursal o línea de producto.
- **Qué buscar**: Clusters (grupos de puntos), alineaciones (relaciones lineales), o puntos aislados (outliers).
""")

# Aplicar filtros
data_filtered = df_filtered.copy()

# Selección de variables para el gráfico 3D
numeric_cols = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]
st.markdown("**🎯 Selecciona tres variables para el gráfico 3D:**")
selected_vars = st.multiselect(
    "Selecciona variables",
    options=numeric_cols,
    default=["Total", "cogs", "Rating"],
    key="vars_filter_section7"
)

if len(selected_vars) != 3:
    st.warning("Por favor, selecciona exactamente tres variables para el gráfico 3D.")
elif data_filtered.empty:
    st.warning("No hay datos disponibles con los filtros seleccionados.")
else:
    # Gráfico de dispersión 3D
    fig7_1 = px.scatter_3d(
        data_filtered,
        x=selected_vars[0],
        y=selected_vars[1],
        z=selected_vars[2],
        size="Quantity",
        hover_data=["Product line", "Branch", "Customer type"],
        title=f"Relación 3D: {selected_vars[0]} vs. {selected_vars[1]} vs. {selected_vars[2]}",
        labels={
            selected_vars[0]: selected_vars[0].replace("_", " ").title(),
            selected_vars[1]: selected_vars[1].replace("_", " ").title(),
            selected_vars[2]: selected_vars[2].replace("_", " ").title()
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=15
    )
    fig7_1.update_layout(
        title_x=0.5,
        scene=dict(
            xaxis_title=selected_vars[0].replace("_", " ").title(),
            yaxis_title=selected_vars[1].replace("_", " ").title(),
            zaxis_title=selected_vars[2].replace("_", " ").title(),
            bgcolor="rgba(0,0,0,0)"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Línea de Producto"
    )

    # Mostrar gráfico 3D
    st.plotly_chart(fig7_1, use_container_width=True)

    # Tabla de correlaciones para las variables seleccionadas
    corr_matrix = data_filtered[selected_vars].corr()
    st.markdown("**Correlaciones entre las variables seleccionadas:**")
    st.dataframe(
        corr_matrix.style.format("{:.2f}"),
        use_container_width=True
    )

# Mapa de calor original (colapsable)
with st.expander("Ver Mapa de Calor de Correlaciones"):
    cols_numericas = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]
    corr_matrix = data_filtered[cols_numericas].corr()

    # Crear mapa de calor con plotly
    fig7_2 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=12),
        hovertemplate="Variable X: %{x}<br>Variable Y: %{y}<br>Correlación: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Correlación")
    ))

    fig7_2.update_layout(
        title="Matriz de Correlación entre Variables Numéricas",
        title_x=0.5,
        xaxis=dict(tickangle=45, title=""),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=700,
        height=600,
        margin=dict(l=100, r=100, t=100, b=100)
    )
     # Mostrar mapa de calor
    st.plotly_chart(fig7_2, use_container_width=True)

# -------------------------------
# 8. Composición del Ingreso Bruto por Sucursal y Línea de Producto
# -------------------------------
st.subheader("8️⃣ Composición del Ingreso Bruto por Sucursal y Línea de Producto")
df_grouped = df_filtered.groupby(["Branch", "Product line"])["gross income"].sum().reset_index()
fig8 = px.bar(df_grouped, x="Branch", y="gross income", color="Product line", barmode="stack",
              title="Ingreso Bruto por Sucursal y Línea de Producto", labels={"gross income": "Ingreso Bruto", "Branch": "Sucursal"})
st.plotly_chart(fig8, use_container_width=True)
# -------------------------------
