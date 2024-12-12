import streamlit as st
import joblib
import pandas as pd
import os
from scipy.sparse import csr_matrix
import re

# Ορισμός της συνάρτησης to_sparse αν την χρειάζεται το pipeline
def to_sparse(X):
    return csr_matrix(X)

# Προσθήκη προσαρμοσμένου CSS για βελτίωση του στυλ
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Διαδρομή στο pipeline (μοντέλο + preprocessing)
model_path = 'models/mlp_model.pkl'  # Χρησιμοποίησε μπροστινή κάθετο ή διπλά backslashes για Windows

# Προσπάθεια φόρτωσης του pipeline
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"⚠️ Σφάλμα κατά τη φόρτωση του pipeline: {str(e)}")
    st.stop()

# Sidebar με ανανεωμένη περιγραφή (χωρίς εικόνα)
st.sidebar.title("📰 Ανίχνευση Ψευδών Ειδήσεων")
st.sidebar.markdown("""
**Καλώς ήρθατε στην εφαρμογή μας για την ανίχνευση ψευδών ειδήσεων!**

Με αυτήν την εφαρμογή μπορείτε εύκολα να ελέγξετε αν ένας τίτλος ειδήσεων είναι **πραγματικός** ή **ψευδής**. Ακολουθήστε τα παρακάτω βήματα:

1. **Εισάγετε τον τίτλο ειδήσεων:** Γράψτε ή επικολλήστε τον τίτλο της είδησης που θέλετε να ελέγξετε στο πεδίο παρακάτω.
2. **Κάντε κλικ στο "🔮 Πρόβλεψη":** Η εφαρμογή θα αναλύσει τον τίτλο χρησιμοποιώντας προηγμένα εργαλεία τεχνητής νοημοσύνης.
3. **Δείτε το αποτέλεσμα:** Θα λάβετε μια απάντηση αν η είδηση είναι **Πραγματική** ή **Ψευδής**, συνοδευόμενη από ένα ποσοστό εμπιστοσύνης που δείχνει πόσο σίγουρη είναι η πρόβλεψη.

**Γιατί να χρησιμοποιήσετε την εφαρμογή μας;**
- **Ασφαλής Πληροφόρηση:** Βοηθά σας να αποφύγετε την παραπληροφόρηση και να λαμβάνετε πιο αξιόπιστες πληροφορίες.
- **Γρήγορη και Απλή Χρήση:** Μόνο λίγα κλικ για να λάβετε μια αξιοπιστία πρόβλεψη.
- **Εκπαιδευμένο Μοντέλο:** Χρησιμοποιούμε τα τελευταία δεδομένα και τεχνικές για να σας προσφέρουμε ακριβείς αποτελέσματα.

**Σημαντικές Σημειώσεις:**
- Η εφαρμογή αναλύει μόνο τον τίτλο της είδησης και δεν λαμβάνει υπόψη το πλήρες περιεχόμενο.
- Η πρόβλεψη βασίζεται σε ιστορικά δεδομένα και δεν αποτελεί απόλυτη διαβεβαίωση για την αλήθεια της είδησης.
- Η εισαγωγή τίτλων ειδήσεων να γραφεί στα Αγγλικά για καλύτερα αποτελέσματα.
""")

# Κύριο περιεχόμενο με το λογότυπο στην κορυφή
st.markdown("<br>", unsafe_allow_html=True)  # Προσθήκη κενής γραμμής για καλύτερη εμφάνιση

# Προσθήκη λογότυπου και τίτλου σε μια γραμμή
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        # Διαδρομή εικόνας για το λογότυπο
        logo_path = "images/Fake News Detection App specializing in political and gossip news.jpg"  # Αντικαταστήστε με την πραγματική διαδρομή της εικόνας σας
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)  # Ρύθμιση του πλάτους του λογότυπου
        else:
            # Χρήση διαδικτυακής εικόνας ως placeholder
            st.image("https://via.placeholder.com/150", width=150)
    with col2:
        st.title("📰 Ανίχνευση Ψευδών Ειδήσεων με MLP")

st.markdown(
    """
    ### Περιγραφή
    Εισάγετε έναν τίτλο ειδήσεων παρακάτω και η εφαρμογή θα προβλέψει αν είναι **Ψευδής** ή **Πραγματικός**.
    
    Το σύστημα χρησιμοποιεί ένα εκπαιδευμένο pipeline που περιλαμβάνει **TF-IDF**, **numeric features**, και **MLP** για την πρόβλεψη.
    """
)

# Εισαγωγή τίτλου με καλύτερη διάταξη
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        title = st.text_input("🔍 Εισάγετε τίτλο ειδήσεων:", "", placeholder="Π.χ., Νέα για την τεχνολογία XYZ")

# Συνάρτηση για προετοιμασία δεδομένων εισόδου
def prepare_input_data(titles):
    data = []
    for t in titles:
        url_length = len(t)  # Χρησιμοποιούμε το μήκος του τίτλου
        contains_fake_word = 1 if "fake" in t.lower() else 0
        data.append({'title': t, 'url_length': url_length, 'contains_fake_word': contains_fake_word})
    return pd.DataFrame(data)

# Συνάρτηση για καθαρισμό εισόδου από χρήστη
def clean_user_input(text):
    # Αφαίρεση επιπλέον κενών και ειδικών χαρακτήρων
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Πρόβλεψη με βελτιωμένο κουμπί και εμφάνιση αποτελεσμάτων
if st.button("🔮 Πρόβλεψη"):
    if title.strip():
        clean_title = clean_user_input(title)
        input_df = prepare_input_data([clean_title])
        # Χρήση spinner για ένδειξη επεξεργασίας
        with st.spinner("Γίνεται επεξεργασία..."):
            try:
                # Πρόβλεψη με το pipeline
                prediction = model.predict(input_df)[0]
                # Υποθέτουμε ότι το μοντέλο υποστηρίζει predict_proba
                probabilities = model.predict_proba(input_df)[0]
                confidence = max(probabilities) * 100

                if prediction == 0:
                    st.error(f"🛑 **Αποτέλεσμα:** Ψευδής Είδηση\n\n🔍 **Βεβαιότητα:** {confidence:.2f}%")
                    st.progress(confidence / 100)
                else:
                    st.success(f"✅ **Αποτέλεσμα:** Πραγματική Είδηση\n\n🔍 **Βεβαιότητα:** {confidence:.2f}%")
                    st.progress(confidence / 100)

            except Exception as e:
                st.error(f"⚠️ Σφάλμα: {str(e)}")
    else:
        st.warning("⚠️ Παρακαλώ εισάγετε έναν τίτλο ειδήσεων.")

# Footer με καλύτερη μορφοποίηση
st.markdown("---")
footer = """
<div style="text-align: center; padding: 10px;">
    <p>🔧 **Προγραμματιστής:** <a href="mailto:Th4nosMylonas@gmail.com" target="_blank">Θανάσης Μυλωνάς</a> | 🌐 **GitHub:** <a href="https://github.com/Th4nosMyl" target="_blank">Th4nosMyl</a></p>
    <p>© 2024 Ανίχνευση Ψευδών Ειδήσεων</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
