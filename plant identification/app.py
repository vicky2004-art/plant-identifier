import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# -----------------------------
# BASIC PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Plant Species Identification (Decision Tree)",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Species Identification System")

st.markdown(
    """
This web app identifies the **plant species** based on:

- **Plant height** (cm)  
- **Leaf width** (cm)  
- **Stem quality** (thin / medium / thick)  

The backend uses a **Decision Tree Classifier in Python (scikit-learn)**.
"""
)

st.markdown("---")

# -----------------------------
# PLANT KNOWLEDGE BASE
# -----------------------------
PLANT_DB = {
    "rose": {
        "name": "Garden Rose (Rosa spp.)",
        "family": "Rosaceae",
        "group": "Flowering shrub",
        "image": "images/rose.jpg",
        "description": """
Roses are woody, perennial flowering shrubs widely grown for their attractive and often fragrant blooms.
They usually have multiple stems bearing thorns, and the leaves are compound with serrated leaflets.
Roses prefer well-drained soil and full sun and are extremely popular in ornamental gardens and as cut flowers.

**Key characteristics:**
- Height: 30–150 cm (dwarf to medium shrubs)
- Stems: woody, often with sharp thorns
- Leaves: compound, 5–7 leaflets, serrated edges
- Flowers: large, layered petals in many colors
- Uses: ornamental gardening, perfumes, rose oil, cut flowers
""",
        "other_plants": [
            "Wild Rose (Rosa canina)",
            "Damask Rose (Rosa × damascena)",
            "China Rose (Rosa chinensis)",
        ],
    },
    "sunflower": {
        "name": "Sunflower (Helianthus annuus)",
        "family": "Asteraceae",
        "group": "Tall annual herb",
        "image": "images/sunflower.jpg",
        "description": """
Sunflowers are tall annual herbs known for their large, bright yellow flower heads.
The flower head is a composite of many small florets. The plant has a thick, fibrous stem
and broad, rough leaves, and is cultivated worldwide for edible seeds and oil.

**Key characteristics:**
- Height: 150–300 cm or more
- Stem: thick, erect, somewhat hairy
- Leaves: broad, heart-shaped, rough texture
- Flower: large yellow head with many florets
- Uses: seeds, cooking oil, ornamental, bird feed
""",
        "other_plants": [
            "Jerusalem Artichoke (Helianthus tuberosus)",
            "Oxeye Daisy (Leucanthemum vulgare)",
            "Marigold (Tagetes spp.)",
        ],
    },
    "lavender": {
        "name": "Lavender (Lavandula angustifolia)",
        "family": "Lamiaceae",
        "group": "Aromatic subshrub",
        "image": "images/lavender.jpg",
        "description": """
Lavender is a small aromatic shrub with narrow, gray-green leaves and violet flower spikes.
It contains essential oils with a strong fragrance and is widely used in perfumery and aromatherapy.

**Key characteristics:**
- Height: 30–90 cm
- Growth form: low, bushy subshrub with woody base
- Leaves: narrow, linear, gray-green
- Flowers: small, violet to purple, in spikes
- Uses: essential oils, perfumes, herbal remedies, ornamental borders
""",
        "other_plants": [
            "Rosemary (Salvia rosmarinus)",
            "Sage (Salvia officinalis)",
            "Thyme (Thymus vulgaris)",
        ],
    },
    "bamboo": {
        "name": "Bamboo (Bambusoideae spp.)",
        "family": "Poaceae (Grasses)",
        "group": "Woody grass",
        "image": "images/bamboo.jpg",
        "description": """
Bamboo is a fast-growing woody grass with hollow, segmented stems called culms.
Many species grow very tall and form dense clumps or groves. Bamboo is an important
material for construction, furniture, paper and also provides edible young shoots.

**Key characteristics:**
- Height: from 2 m to over 20 m (depending on species)
- Stem: hollow, jointed culms with prominent nodes
- Leaves: elongated, lance-shaped
- Growth: clumping or running, forming thickets
- Uses: building material, furniture, flooring, paper, edible shoots
""",
        "other_plants": [
            "Sugarcane (Saccharum officinarum)",
            "Common Reed (Phragmites australis)",
            "Giant Reed (Arundo donax)",
        ],
    },
}

STEM_MAP = {"thin": 0, "medium": 1, "thick": 2}

# -----------------------------
# TRAIN DECISION TREE
# -----------------------------
def train_model():
    X = []
    y = []

    # Lavender: short / medium, thin stem, narrow leaves
    lavender_samples = [
        [30, 1.0, STEM_MAP["thin"]],
        [40, 1.5, STEM_MAP["thin"]],
        [60, 2.0, STEM_MAP["thin"]],
        [50, 2.3, STEM_MAP["thin"]],
    ]
    X.extend(lavender_samples)
    y.extend(["lavender"] * len(lavender_samples))

    # Rose: medium height shrubs, thin/medium stem, moderate width
    rose_samples = [
        [60, 4.0, STEM_MAP["thin"]],
        [80, 5.0, STEM_MAP["medium"]],
        [120, 6.0, STEM_MAP["medium"]],
        [90, 3.5, STEM_MAP["thin"]],
    ]
    X.extend(rose_samples)
    y.extend(["rose"] * len(rose_samples))

    # Sunflower: tall, medium/thick stem, broad leaves
    sunflower_samples = [
        [180, 10.0, STEM_MAP["medium"]],
        [200, 12.0, STEM_MAP["thick"]],
        [220, 8.0, STEM_MAP["medium"]],
        [250, 11.0, STEM_MAP["thick"]],
    ]
    X.extend(sunflower_samples)
    y.extend(["sunflower"] * len(sunflower_samples))

    # Bamboo: tall to very tall, thick stem, narrower leaves
    bamboo_samples = [
        [220, 3.0, STEM_MAP["thick"]],
        [300, 4.0, STEM_MAP["thick"]],
        [260, 2.5, STEM_MAP["thick"]],
        [350, 5.0, STEM_MAP["thick"]],
    ]
    X.extend(bamboo_samples)
    y.extend(["bamboo"] * len(bamboo_samples))

    X = np.array(X)
    y = np.array(y)

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model()

def classify_plant(height_cm, width_cm, stem_quality):
    stem_code = STEM_MAP[stem_quality]
    sample = np.array([[height_cm, width_cm, stem_code]])
    pred = model.predict(sample)[0]
    rules = export_text(
        model,
        feature_names=["height_cm", "leaf_width_cm", "stem_quality_code"],
    )
    return pred, rules

# -----------------------------
# UI LAYOUT
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("🔢 Input plant measurements")

    height = st.slider(
        "Plant height (cm)",
        min_value=10.0,
        max_value=400.0,
        value=80.0,
        step=1.0,
    )
    width = st.slider(
        "Leaf width (cm)",
        min_value=0.2,
        max_value=20.0,
        value=4.0,
        step=0.1,
    )
    stem_quality = st.selectbox(
        "Stem quality",
        options=list(STEM_MAP.keys()),
        index=1,
    )

    st.caption(
        f"Selected: height = {height:.1f} cm, leaf width = {width:.1f} cm, stem = {stem_quality}"
    )

    identify = st.button("Identify species")

with right:
    st.subheader("✅ Identification result")

    if identify:
        species_key, rules_text = classify_plant(height, width, stem_quality)
        plant = PLANT_DB[species_key]

        st.success(f"Predicted species: **{plant['name']}**")
        st.write(f"**Family:** {plant['family']} · **Group:** {plant['group']}")

        # Image (optional)
        try:
            st.image(plant["image"], caption=plant["name"], use_column_width=True)
        except Exception:
            st.warning("Image not found. Put an image file in the `images/` folder.")

        st.markdown("#### 🌱 Species description")
        st.markdown(plant["description"])

        st.markdown("#### 🌿 Other plants in this group")
        st.write(", ".join(plant["other_plants"]))

        with st.expander("📊 View decision tree rules"):
            st.text(rules_text)
    else:
        st.info("Click **Identify species** after choosing the measurements.")

st.markdown("---")
st.subheader("📚 Plant species included in this demo")
cols = st.columns(4)
for i, (k, plant) in enumerate(PLANT_DB.items()):
    with cols[i % 4]:
        st.markdown(f"**{plant['name']}**")
        st.caption(f"{plant['family']} · {plant['group']}")
