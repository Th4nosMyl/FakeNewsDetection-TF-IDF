# 📰 Fake News Detection Application

![Logo](images/Fake%20News%20Detection%20App%20specializing%20in%20political%20and%20gossip%20news.jpg)

## 📚 Εισαγωγή

Στον σύγχρονο ψηφιακό κόσμο, η πληροφορία διαδίδεται ταχύτατα μέσω των κοινωνικών δικτύων, των ειδησεογραφικών ιστοσελίδων και άλλων ψηφιακών πλατφορμών. Ωστόσο, μαζί με την αυξανόμενη διαθεσιμότητα της πληροφορίας, παρατηρείται και η αύξηση των ψευδών ειδήσεων (fake news), οι οποίες μπορούν να προκαλέσουν παραπληροφόρηση, κοινωνική αναστάτωση και ακόμα και πολιτικές επιπτώσεις.

## 🎯 Στόχος του Project

Ο κύριος στόχος αυτού του project είναι η ανάπτυξη μιας εφαρμογής που θα επιτρέπει στους χρήστες να ελέγχουν αν ένας τίτλος ειδήσεων είναι **πραγματικός** ή **ψευδής**. Με αυτόν τον τρόπο, επιδιώκουμε να συμβάλουμε στην αντιμετώπιση της παραπληροφόρησης και να ενισχύσουμε την αξιοπιστία των πληροφοριών που διαδίδονται.

## 🛠️ Τεχνολογίες που Χρησιμοποιήθηκαν

- **Python:** Η κύρια γλώσσα προγραμματισμού που χρησιμοποιήθηκε για την ανάπτυξη του project.
- **Streamlit:** Ένα framework που διευκολύνει τη δημιουργία διαδραστικών εφαρμογών ιστού για machine learning.
- **Scikit-learn:** Βιβλιοθήκη για την εφαρμογή αλγορίθμων μηχανικής μάθησης.
- **XGBoost:** Μια ισχυρή βιβλιοθήκη για boosting αλγορίθμους που χρησιμοποιείται για την εκπαίδευση μοντέλων πρόβλεψης.
- **Joblib:** Εργαλείο για την αποθήκευση και φόρτωση μοντέλων μηχανικής μάθησης.
- **Pandas:** Βιβλιοθήκη για την επεξεργασία και ανάλυση δεδομένων.

## 📊 Dataset: FakeNewsNet

### Τι είναι το FakeNewsNet;

Το **FakeNewsNet** είναι ένα δημόσιο διαθέσιμο σύνολο δεδομένων που χρησιμοποιείται για την εκπαίδευση και αξιολόγηση μοντέλων ανίχνευσης ψευδών ειδήσεων. Περιλαμβάνει πραγματικές και ψευδείς ειδήσεις που έχουν συλλεχθεί από διάφορες πηγές, προσφέροντας έτσι μια πλούσια βάση για την ανάπτυξη αλγορίθμων μηχανικής μάθησης.

### Γιατί το FakeNewsNet;

- **Πλούσιο και Διαφορετικό Περιεχόμενο:** Περιλαμβάνει ειδήσεις από διάφορες κατηγορίες και πηγές, εξασφαλίζοντας έτσι την ποικιλία και την αντιπροσωπευτικότητα των δεδομένων.
- **Ποιοτικά Στοιχεία:** Περιλαμβάνει τόσο το περιεχόμενο των ειδήσεων όσο και μεταδεδομένα που βοηθούν στην ανάλυση και την πρόβλεψη.
- **Δημόσια Διαθεσιμότητα:** Είναι ελεύθερα διαθέσιμο για έρευνα και ανάπτυξη, καθιστώντας το ιδανικό για ακαδημαϊκά και επαγγελματικά project.

## 🔧 Προεπεξεργασία Δεδομένων

Η προεπεξεργασία των δεδομένων είναι ένα κρίσιμο βήμα για την επιτυχία των μοντέλων μηχανικής μάθησης. Στην εφαρμογή μας, ακολουθήσαμε τα εξής βήματα:

1. **Καθαρισμός Κειμένου:**
   - Αφαίρεση ειδικών χαρακτήρων, αριθμών και περιττών κενών.
   - Μετατροπή όλων των χαρακτήρων σε μικρά γράμματα για ομοιομορφία.

2. **Tokenization:**
   - Διαχωρισμός του κειμένου σε λέξεις (tokens) για ευκολότερη επεξεργασία.

3. **Stop Words Removal:**
   - Αφαίρεση κοινών λέξεων (όπως "και", "ή", "σε") που δεν προσθέτουν σημαντική πληροφορία για την ταξινόμηση.

4. **Stemming και Lemmatization:**
   - Μείωση των λέξεων στη ρίζα τους για τη μείωση της πολυμορφίας του λεξιλογίου.

5. **Δημιουργία Χαρακτηριστικών:**
   - **TF-IDF Vectorization:** Μετατροπή του κειμένου σε αριθμητικά χαρακτηριστικά που αντικατοπτρίζουν τη σημαντικότητα των λέξεων.
   - **Numeric Features:** Δημιουργία επιπλέον χαρακτηριστικών όπως το μήκος του τίτλου και η παρουσία συγκεκριμένων λέξεων (π.χ., "fake").

### Γιατί αυτή η Προεπεξεργασία;

- **Απλότητα και Απόδοση:** Οι τεχνικές αυτές βοηθούν στη μείωση της πολυπλοκότητας των δεδομένων και στην ενίσχυση των σημαντικών πληροφοριών.
- **Βελτιστοποίηση Μοντέλων:** Με καθαρά και καλά προετοιμασμένα δεδομένα, τα μοντέλα μπορούν να εκπαιδευτούν πιο αποτελεσματικά και να επιτύχουν υψηλότερη ακρίβεια.
- **Ελαχιστοποίηση Θορύβου:** Η αφαίρεση περιττών στοιχείων μειώνει τον θόρυβο στα δεδομένα, καθιστώντας την πρόβλεψη πιο αξιόπιστη.

## 🤖 Αλγόριθμοι που Χρησιμοποιήθηκαν

Για την ανίχνευση ψευδών ειδήσεων, χρησιμοποιήσαμε τέσσερις διαφορετικούς αλγορίθμους μηχανικής μάθησης. Κάθε ένας από αυτούς έχει τα δικά του πλεονεκτήματα και επιλέχθηκαν για να εξασφαλίσουν την καλύτερη δυνατή απόδοση του συστήματος μας.

### 1. Support Vector Machine (SVM)

- **Τι είναι το SVM;**
  Το SVM είναι ένας επιβλεπόμενος αλγόριθμος ταξινόμησης που προσπαθεί να βρει την υπέρτατη υπερεπίπεδη γραμμή ή επιφάνεια που διαχωρίζει τα δεδομένα σε διαφορετικές κατηγορίες.

- **Γιατί το SVM;**
  - **Υψηλή Απόδοση:** Εξαιρετική απόδοση σε μικρά και μεσαία σύνολα δεδομένων.
  - **Ανθεκτικότητα στον Θόρυβο:** Καλή διαχείριση μεθόδων μετασχηματισμού των δεδομένων, καθιστώντας το ανθεκτικό σε θορύβους και ατελείς πληροφορίες.

### 2. Random Forest

- **Τι είναι το Random Forest;**
  Το Random Forest είναι ένα ensemble learning method που βασίζεται σε πολλαπλά δέντρα αποφάσεων. Κάθε δέντρο εκπαιδεύεται σε τυχαία υποσύνολα των δεδομένων και οι προβλέψεις τους συνδυάζονται για την τελική απόφαση.

- **Γιατί το Random Forest;**
  - **Ευελιξία:** Ικανό να χειρίζεται τόσο κατηγοριοποιημένα όσο και συνεχή χαρακτηριστικά.
  - **Αντιμετώπιση Overfitting:** Μέσω της τυχαίας επιλογής χαρακτηριστικών και δειγμάτων, μειώνει την πιθανότητα υπερπροσαρμογής.

### 3. XGBoost

- **Τι είναι το XGBoost;**
  Το XGBoost (Extreme Gradient Boosting) είναι μια βελτιστοποιημένη εκδοχή του gradient boosting, η οποία χρησιμοποιείται για την οικοδόμηση ισχυρών μοντέλων πρόβλεψης.

- **Γιατί το XGBoost;**
  - **Υψηλή Απόδοση:** Συνήθως υπερτερεί σε πολλαπλά datasets λόγω της ικανότητάς του να μαθαίνει σύνθετες σχέσεις.
  - **Κλίμακα:** Αποδοτικό σε μεγάλες ποσότητες δεδομένων και παράλληλη επεξεργασία.

### 4. Multilayer Perceptron (MLP)

- **Τι είναι το MLP;**
  Το MLP είναι ένας τύπος νευρωνικού δικτύου που αποτελείται από πολλαπλά στρώματα νευρώνων. Χρησιμοποιείται για την επίλυση προβλημάτων ταξινόμησης και παλινδρόμησης.

- **Γιατί το MLP;**
  - **Ευελιξία:** Ικανό να μάθει μη γραμμικές σχέσεις μεταξύ των χαρακτηριστικών.
  - **Ικανότητα Προσαρμογής:** Μπορεί να προσαρμοστεί σε πολύπλοκα πρότυπα δεδομένων.

## 🌟 Χαρακτηριστικά της Εφαρμογής

- **Εύκολη Χρήση:** Απλό περιβάλλον όπου οι χρήστες μπορούν να εισάγουν έναν τίτλο ειδήσεων και να λάβουν άμεσα την πρόβλεψη.
- **Αποτελεσματική Πρόβλεψη:** Χρήση προηγμένων αλγορίθμων μηχανικής μάθησης για ακριβείς προβλέψεις.
- **Ποσοστό Εμπιστοσύνης:** Παροχή ενός ποσοστού εμπιστοσύνης που δείχνει πόσο σίγουρη είναι η πρόβλεψη του μοντέλου.
- **Οπτική Παρουσίαση:** Ελκυστική και καθαρή διεπαφή χρήστη με λογότυπο και ευανάγνωστο περιεχόμενο.

## 🚀 Οφέλη του Project

- **Μείωση Παραπληροφόρησης:** Βοηθά τους χρήστες να εντοπίζουν και να αποφεύγουν την παραπληροφόρηση.
- **Ενίσχυση Κριτικής Σκέψης:** Ενθαρρύνει τους χρήστες να αξιολογούν την αξιοπιστία των ειδήσεων που διαβάζουν.
- **Εκπαιδευτική Αξία:** Παρέχει ένα πρακτικό παράδειγμα εφαρμογής μηχανικής μάθησης στην αντιμετώπιση κοινωνικών προβλημάτων.

## 📄 Πώς να Ξεκινήσετε

Για να ξεκινήσετε με την εφαρμογή, ακολουθήστε τα παρακάτω βήματα:

1. **Εγκατάσταση Απαραίτητων Βιβλιοθηκών:**
   Βεβαιωθείτε ότι έχετε εγκαταστήσει όλες τις απαραίτητες βιβλιοθήκες Python που αναφέρονται στο [requirements.txt](requirements.txt).

2. **Εκτέλεση της Εφαρμογής:**
   Χρησιμοποιήστε την εντολή `streamlit run app.py` στο τερματικό σας για να τρέξετε την εφαρμογή τοπικά.

3. **Χρήση της Εφαρμογής:**
   Εισάγετε έναν τίτλο ειδήσεων στο πεδίο εισαγωγής και κάντε κλικ στο "🔮 Πρόβλεψη" για να λάβετε την πρόβλεψη.

## 📊 Παραδείγματα Χρήσης

- **Τίτλος:** "Breaking: new vaccine discovered!"
  - **Πρόβλεψη:** ✅ **Πραγματική Είδηση**
  - **Βεβαιότητα:** 92.50%

- **Τίτλος:** "Fake News: Professor XYZ reveals the truth!"
  - **Πρόβλεψη:** 🛑 **Ψευδής Είδηση**
  - **Βεβαιότητα:** 85.30%

- **Τίτλος:** "New education policy announced"
  - **Πρόβλεψη:** ✅ **Πραγματική Είδηση**
  - **Βεβαιότητα:** 88.40%

## 📧 Επικοινωνία

Για περισσότερες πληροφορίες ή ερωτήσεις σχετικά με το project, μπορείτε να επικοινωνήσετε με τον προγραμματιστή:

- **Προγραμματιστής:** [Θανάσης Μυλωνάς](mailto:Th4nosMylonas@gmail.com)
- **GitHub:** [Th4nosMyl](https://github.com/Th4nosMyl)