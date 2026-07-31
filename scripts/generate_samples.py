"""
scripts/generate_samples.py
Genere des donnees financieres fictives realistes pour tester le pipeline RAG.

Produit dans data/raw/ :
    - rapport_fraude_2022.txt, rapport_fraude_2023.txt, rapport_fraude_2024.txt
    - transactions_suspectes.csv     (1000 lignes)
    - alertes_conformite.json        (200 alertes)
    - guide_detection_fraude.txt
    - glossaire_termes_fraude.txt
    - cas/cas_0001.txt ... cas_0040.txt   (40 recits de cas anonymises)
"""
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

random.seed(42)
config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
CAS_DIR = config.DATA_RAW_DIR / "cas"
CAS_DIR.mkdir(parents=True, exist_ok=True)


def random_date(start_year=2023):
    start = datetime(start_year, 1, 1)
    return (start + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d %H:%M")


def random_iban():
    return f"FR{random.randint(10,99)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(100,999)}"


# ── 1. Rapports annuels narratifs (2022, 2023, 2024) ──────────────────────────

RAPPORT_2023 = """
RAPPORT ANNUEL — DETECTION DE FRAUDE ET GESTION DES RISQUES
Exercice 2023 — Direction de la Conformite et du Controle Interne

========================================================================
SYNTHESE EXECUTIVE
========================================================================

Au cours de l'exercice 2023, le departement de detection de fraude a traite
un total de 14 872 alertes, dont 1 203 ont donne lieu a une investigation
approfondie. Sur ce perimetre, 287 cas de fraude averee ont ete confirmes,
representant un prejudice total estime a 4,7 millions d'euros.

Le taux de fraude confirmee rapporte au volume total de transactions s'etablit
a 0,003%, en legere hausse par rapport a 2022 (0,0024%) mais en deca des
moyennes sectorielles europeennes (0,006%).

========================================================================
CHAPITRE 1 — TYPOLOGIES DE FRAUDE IDENTIFIEES
========================================================================

1.1 FRAUDE PAR USURPATION D'IDENTITE (38% des cas)

La fraude par usurpation d'identite represente la menace principale.
Les schemas les plus frequents observes en 2023 incluent :

- Ouverture frauduleuse de comptes avec des pieces d'identite contrefaites (22%)
- Prise de controle de compte (Account Takeover) via phishing (16%)
- Fraude documentaire lors des demandes de credit (12%)

Montant moyen par dossier : 8 400 EUR
Delai moyen de detection : 12 jours

1.2 FRAUDE AUX VIREMENTS (29% des cas)

Les fraudes aux virements ont augmente de 34% par rapport a 2022, portees par
l'essor des arnaques au faux conseiller bancaire.

Patterns detectes :
- Virement vers compte mule avec fractionnement (smurfing)
- Virement international atypique hors zones habituelles du client
- Modification soudaine du beneficiaire habituel

Montant moyen : 23 500 EUR
Delai de detection : 6 jours (amelioration grace a l'IA temps reel)

1.3 FRAUDE INTERNE (18% des cas)

La fraude interne reste un risque significatif, principalement portee par :
- Detournement de commissions (43% des fraudes internes)
- Modification non autorisee de donnees clients (31%)
- Acces illicite a des comptes premium (26%)

Secteurs les plus touches : Gestion patrimoniale, Credit entreprise

1.4 FRAUDE AUX CHEQUES ET FAUX BONS DE CAISSE (15% des cas)

En recul structurel du fait de la dematerialisation, mais encore presente
dans les segments clientele senior.

========================================================================
CHAPITRE 2 — INDICATEURS D'ALERTE (RED FLAGS)
========================================================================

Les algorithmes de detection s'appuient sur les signaux suivants :

SIGNAUX FORTS (score > 80) :
- Transaction superieure a 3 fois la moyenne historique du compte
- Connexion depuis un pays non habituellement frequente
- Plusieurs tentatives echouees suivies d'un succes
- Modification d'adresse + virement dans les 24h
- Virement vers compte cree il y a moins de 30 jours

SIGNAUX MODERES (score 50-80) :
- Transaction a une heure inhabituelle (entre 01h et 05h)
- Virement fractionne en plusieurs operations similaires
- Utilisation d'un nouveau dispositif d'authentification
- Transaction dans un secteur inhabituel (ex: casino pour client sans historique)

SIGNAUX FAIBLES (score 20-50) :
- Augmentation soudaine de la frequence de transactions
- Geolocalisation incoherente avec l'adresse de facturation
- Utilisation d'un VPN ou IP masquee

========================================================================
CHAPITRE 3 — PERFORMANCE DES MODELES DE DETECTION
========================================================================

Trois modeles sont en production en 2023 :

MODELE A — XGBoost Transactions (deploye Q1 2022)
- Precision : 94,2%
- Rappel : 87,6%
- F1-score : 90,8%
- Faux positifs/jour : 142

MODELE B — LSTM Comportemental (deploye Q3 2023)
- Precision : 96,1%
- Rappel : 91,3%
- F1-score : 93,6%
- Faux positifs/jour : 89 (-37% vs Modele A)

MODELE C — Graph Neural Network Relations (pilote Q4 2023)
- En phase de validation sur 5% du flux
- Detection de reseaux de fraude organisee prometteuse
- Resultats preliminaires : +18% de detection des schemas coordonnes

========================================================================
CHAPITRE 4 — RECOMMANDATIONS 2024
========================================================================

1. Deploiement complet du Modele B sur 100% du flux transactionnel
2. Integration du GNN en production pour la detection de reseaux
3. Renforcement de l'authentification forte pour les virements > 10 000 EUR
4. Formation des equipes de back-office sur les nouvelles typologies de fraude
5. Partenariat avec 3 autres etablissements pour partage des listes noires

Directeur de la Conformite : Jean-Pierre MARTIN
Date : 15 janvier 2024
""".strip()

RAPPORT_2022 = """
RAPPORT ANNUEL — DETECTION DE FRAUDE ET GESTION DES RISQUES
Exercice 2022 — Direction de la Conformite et du Controle Interne

========================================================================
SYNTHESE EXECUTIVE
========================================================================

Au cours de l'exercice 2022, le departement de detection de fraude a traite
un total de 12 340 alertes, dont 980 ont donne lieu a une investigation
approfondie. Sur ce perimetre, 214 cas de fraude averee ont ete confirmes,
representant un prejudice total estime a 3,1 millions d'euros.

Le taux de fraude confirmee s'etablit a 0,0024%, stable par rapport a 2021
(0,0023%) et en deca des moyennes sectorielles europeennes (0,006%).

========================================================================
CHAPITRE 1 — TYPOLOGIES DE FRAUDE IDENTIFIEES
========================================================================

1.1 FRAUDE PAR USURPATION D'IDENTITE (41% des cas)

Deja premiere typologie en 2022, avec une part legerement superieure a 2023 :

- Ouverture frauduleuse de comptes avec des pieces d'identite contrefaites (25%)
- Prise de controle de compte (Account Takeover) via phishing (11%)
- Fraude documentaire lors des demandes de credit (12%)

Montant moyen par dossier : 7 100 EUR
Delai moyen de detection : 18 jours

1.2 FRAUDE AUX VIREMENTS (22% des cas)

En nette progression sur l'exercice suivant (+34% en 2023), les fraudes aux
virements representaient encore une part modeste en 2022.

Montant moyen : 17 800 EUR
Delai de detection : 9 jours

1.3 FRAUDE INTERNE (21% des cas)

- Detournement de commissions (39% des fraudes internes)
- Modification non autorisee de donnees clients (35%)
- Acces illicite a des comptes premium (26%)

1.4 FRAUDE AUX CHEQUES ET FAUX BONS DE CAISSE (16% des cas)

========================================================================
CHAPITRE 2 — INDICATEURS D'ALERTE (RED FLAGS)
========================================================================

Version initiale des regles de detection, affinees en 2023 :

SIGNAUX FORTS (score > 80) :
- Transaction superieure a 3 fois la moyenne historique du compte
- Connexion depuis un pays non habituellement frequente
- Plusieurs tentatives echouees suivies d'un succes

SIGNAUX MODERES (score 50-80) :
- Transaction a une heure inhabituelle
- Virement fractionne en plusieurs operations similaires

========================================================================
CHAPITRE 3 — PERFORMANCE DES MODELES DE DETECTION
========================================================================

MODELE A — XGBoost Transactions (deploye Q1 2022)
- Precision : 91,8%
- Rappel : 84,2%
- F1-score : 87,9%
- Faux positifs/jour : 203

========================================================================
CHAPITRE 4 — RECOMMANDATIONS 2023
========================================================================

1. Deploiement d'un second modele comportemental (LSTM) pour reduire les
   faux positifs du Modele A
2. Renforcement de la surveillance des virements internationaux
3. Lancement d'un pilote de detection de reseaux de fraude organisee

Directeur de la Conformite : Jean-Pierre MARTIN
Date : 20 janvier 2023
""".strip()

RAPPORT_2024 = """
RAPPORT ANNUEL — DETECTION DE FRAUDE ET GESTION DES RISQUES
Exercice 2024 — Direction de la Conformite et du Controle Interne

========================================================================
SYNTHESE EXECUTIVE
========================================================================

Au cours de l'exercice 2024, le departement de detection de fraude a traite
un total de 17 605 alertes, dont 1 489 ont donne lieu a une investigation
approfondie. Sur ce perimetre, 331 cas de fraude averee ont ete confirmes,
representant un prejudice total estime a 5,9 millions d'euros.

Le taux de fraude confirmee s'etablit a 0,0033%, en hausse mesuree par
rapport a 2023 (0,003%), portee par le deploiement complet du Modele B et
l'integration du GNN qui ameliorent la detection sans faire croitre le
volume de fraude reelle au meme rythme.

========================================================================
CHAPITRE 1 — TYPOLOGIES DE FRAUDE IDENTIFIEES
========================================================================

1.1 FRAUDE PAR USURPATION D'IDENTITE (34% des cas)

Part en recul relatif grace au renforcement de l'authentification forte
recommande fin 2023 :

- Ouverture frauduleuse de comptes avec des pieces d'identite contrefaites (18%)
- Prise de controle de compte (Account Takeover) via phishing (10%)
- Fraude documentaire lors des demandes de credit (6%)

Montant moyen par dossier : 9 200 EUR
Delai moyen de detection : 8 jours (amelioration continue)

1.2 FRAUDE AUX VIREMENTS (33% des cas)

Devient la premiere typologie en volume, portee par la sophistication
croissante des arnaques au faux conseiller et l'usage de deepfakes vocaux
dans 4% des cas detectes.

Montant moyen : 27 100 EUR
Delai de detection : 4 jours

1.3 FRAUDE INTERNE (16% des cas)

1.4 FRAUDE AUX CHEQUES ET FAUX BONS DE CAISSE (12% des cas)

Poursuite du recul structurel.

1.5 FRAUDE PAR RESEAUX ORGANISES (5% des cas, nouvelle categorie)

Premiere annee de suivi distinct grace au Modele C (GNN), desormais en
production complete.

========================================================================
CHAPITRE 2 — INDICATEURS D'ALERTE (RED FLAGS)
========================================================================

SIGNAUX FORTS (score > 80) :
- Transaction superieure a 3 fois la moyenne historique du compte
- Connexion depuis un pays non habituellement frequente
- Plusieurs tentatives echouees suivies d'un succes
- Modification d'adresse + virement dans les 24h
- Virement vers compte cree il y a moins de 30 jours
- Correspondance avec un noeud identifie par le GNN comme reseau de fraude

SIGNAUX MODERES (score 50-80) :
- Transaction a une heure inhabituelle (entre 01h et 05h)
- Virement fractionne en plusieurs operations similaires
- Utilisation d'un nouveau dispositif d'authentification

========================================================================
CHAPITRE 3 — PERFORMANCE DES MODELES DE DETECTION
========================================================================

MODELE B — LSTM Comportemental (100% du flux depuis mars 2024)
- Precision : 96,8%
- Rappel : 92,7%
- F1-score : 94,7%
- Faux positifs/jour : 71

MODELE C — Graph Neural Network Relations (production complete depuis juin 2024)
- Precision : 89,4%
- Rappel : 78,3%
- Reseaux de fraude organisee identifies : 23 (vs 4 en phase pilote 2023)

========================================================================
CHAPITRE 4 — RECOMMANDATIONS 2025
========================================================================

1. Etude de faisabilite d'un quatrieme modele dedie a la detection de
   deepfakes vocaux dans les appels au service client
2. Extension du partage de listes noires a 6 etablissements partenaires
3. Revision du seuil de declaration TRACFIN automatique pour les virements
   internationaux vers pays a risque GAFI

Directeur de la Conformite : Jean-Pierre MARTIN
Date : 17 janvier 2025
""".strip()

for year, content in [(2022, RAPPORT_2022), (2023, RAPPORT_2023), (2024, RAPPORT_2024)]:
    (config.DATA_RAW_DIR / f"rapport_fraude_{year}.txt").write_text(content, encoding="utf-8")
    print(f"- rapport_fraude_{year}.txt")


# ── 2. Transactions suspectes (CSV, 1000 lignes) ──────────────────────────────

PAYS = ["France", "Allemagne", "Espagne", "Italie", "Portugal", "Roumanie", "Nigeria", "Chine", "Ukraine"]
TYPES = ["Virement SEPA", "Virement international", "Paiement carte", "Retrait DAB", "Virement instantane"]
STATUTS = ["ALERTE_HAUTE", "ALERTE_MOYENNE", "ALERTE_FAIBLE", "BLOQUEE", "EN_INVESTIGATION"]
MOTIFS = [
    "Montant anormalement eleve",
    "Pays de destination inhabituel",
    "Heure de transaction suspecte",
    "Fractionnement detecte (smurfing)",
    "Beneficiaire jamais utilise",
    "Vitesse de transactions anormale",
    "Geolocalisation incoherente",
    "Compte beneficiaire recent (<30j)",
    "Modification d'adresse recente",
]

N_TRANSACTIONS = 1000
rows = []
for i in range(N_TRANSACTIONS):
    annee = random.choice([2022, 2023, 2024])
    montant = round(random.choices(
        [random.uniform(500, 5000), random.uniform(5000, 50000), random.uniform(50000, 500000)],
        weights=[60, 30, 10]
    )[0], 2)
    rows.append({
        "id_alerte":        f"ALT-{annee}-{10000 + i}",
        "date_transaction": random_date(start_year=annee),
        "client_id":        f"CLI-{random.randint(100000, 999999)}",
        "type_transaction": random.choice(TYPES),
        "montant_eur":      montant,
        "pays_destination": random.choice(PAYS),
        "iban_beneficiaire": random_iban(),
        "score_fraude":     random.randint(45, 99),
        "statut":           random.choice(STATUTS),
        "motif_alerte":     random.choice(MOTIFS),
        "analyste":         random.choice(["M. Dubois", "S. Chen", "A. Kowalski", "N. Traore", None]),
        "fraude_confirmee": random.choice([True, False, False, False]),
    })

csv_path = config.DATA_RAW_DIR / "transactions_suspectes.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"- transactions_suspectes.csv ({N_TRANSACTIONS} lignes)")


# ── 3. Alertes conformite (JSON, 200 alertes) ─────────────────────────────────

REGLEMENTS = ["LCB-FT", "RGPD", "MIF2", "DSP2", "DORA", "Bale III"]
NIVEAUX    = ["CRITIQUE", "MAJEUR", "MINEUR", "INFORMATIF"]
DESCRIPTIONS_CONF = [
    "Depassement du seuil de declaration TRACFIN sans signalement",
    "Defaut de verification KYC pour client a risque eleve",
    "Absence de gel des avoirs pour client liste UE",
    "Transaction suspecte non declaree dans les delais reglementaires",
    "Utilisation de donnees personnelles hors consentement",
    "Authentification forte non appliquee pour virement > 30 000 EUR",
    "Dossier de due diligence renforcee (EDD) incomplet",
    "Absence de revue periodique du profil de risque client",
]

N_ALERTES = 200
alertes = []
for i in range(N_ALERTES):
    annee = random.choice([2022, 2023, 2024])
    alertes.append({
        "id":           f"CONF-{annee}{i:04d}",
        "date":         random_date(start_year=annee),
        "reglement":    random.choice(REGLEMENTS),
        "niveau":       random.choice(NIVEAUX),
        "description":  random.choice(DESCRIPTIONS_CONF),
        "entite_concernee": random.choice(["Agence Paris 8e", "Direction Credit", "Service Titres", "Banque Privee"]),
        "action_requise": random.choice([
            "Declaration immediate a TRACFIN",
            "Gel preventif du compte",
            "Revision de la procedure KYC",
            "Audit interne dans les 30 jours",
        ]),
        "cloture": random.choice([True, False]),
    })

json_path = config.DATA_RAW_DIR / "alertes_conformite.json"
json_path.write_text(json.dumps(alertes, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"- alertes_conformite.json ({N_ALERTES} alertes)")


# ── 4. Guide de detection ─────────────────────────────────────────────────────

GUIDE = """
GUIDE OPERATIONNEL — DETECTION ET TRAITEMENT DE LA FRAUDE
Version 3.2 — Usage interne

========================================================================
PROCEDURES D'INVESTIGATION
========================================================================

ETAPE 1 : QUALIFICATION DE L'ALERTE

Des reception d'une alerte systeme, l'analyste doit :
a) Verifier le score de fraude : si < 50, classer en surveillance passive
b) Consulter l'historique transactionnel des 90 derniers jours
c) Verifier si le client figure sur une liste de surveillance (OFAC, UE, ONU)
d) Contacter le client par telephone si score > 80 et montant > 5 000 EUR

Delai maximum de qualification : 4 heures ouvrees

ETAPE 2 : BLOCAGE PREVENTIF

Conditions de blocage immediat (sans validation client) :
- Score fraude > 95
- Transaction > 100 000 EUR vers pays a risque GAFI
- Pattern smurfing detecte sur 5 transactions ou plus
- Client identifie sur une liste de sanctions

ETAPE 3 : DECLARATION TRACFIN

Obligation legale de declaration si :
- Fraude averee > 5 000 EUR
- Soupcon de blanchiment d'argent
- Financement du terrorisme suspecte

Delai legal : 15 jours ouvres (recommande : 48h)
Formulaire : TRACFIN-DS (portail securise)

ETAPE 4 : CLOTURE DU DOSSIER

Chaque dossier doit etre clos avec :
- Resume des faits (500 mots minimum)
- Decision motivee (fraude averee / non averee / suspendue)
- Montant recupere ou passe en perte
- Recommandations pour eviter la recurrence

Conservation des dossiers : 10 ans (obligation legale)

========================================================================
CONTACT URGENCES
========================================================================

Cellule fraude 24h/24 : +33 1 XX XX XX XX
Email securise : fraude-urgence@[banque].fr
TRACFIN : https://www.tracfin.gouv.fr
""".strip()

(config.DATA_RAW_DIR / "guide_detection_fraude.txt").write_text(GUIDE, encoding="utf-8")
print("- guide_detection_fraude.txt")


# ── 5. Glossaire des termes metier (fraude / AML / KYC) ───────────────────────

GLOSSAIRE_TERMES = [
    ("TRACFIN", "Traitement du Renseignement et Action contre les Circuits "
     "Financiers clandestins. Service de renseignement financier francais "
     "charge de la lutte contre le blanchiment d'argent et le financement "
     "du terrorisme (LCB-FT). Les etablissements financiers ont l'obligation "
     "legale de lui declarer toute transaction suspecte."),
    ("GAFI", "Groupe d'Action Financiere (FATF en anglais). Organisme "
     "intergouvernemental qui etablit les normes internationales de lutte "
     "contre le blanchiment de capitaux et le financement du terrorisme, et "
     "publie une liste de pays a risque."),
    ("LCB-FT", "Lutte Contre le Blanchiment de capitaux et le Financement du "
     "Terrorisme. Cadre reglementaire regroupant l'ensemble des obligations "
     "des etablissements financiers en matiere de prevention de la fraude "
     "financiere."),
    ("KYC", "Know Your Customer. Ensemble des procedures d'identification et "
     "de verification de l'identite d'un client avant l'entree en relation "
     "d'affaires, incluant la collecte de justificatifs et l'evaluation du "
     "profil de risque."),
    ("EDD", "Enhanced Due Diligence (due diligence renforcee). Verifications "
     "approfondies appliquees aux clients presentant un risque eleve "
     "(personnes politiquement exposees, pays a risque, activites sensibles)."),
    ("PEP", "Personne Politiquement Exposee. Individu occupant ou ayant "
     "occupe une fonction publique importante, soumis a une surveillance "
     "renforcee du fait d'un risque accru de corruption ou de trafic "
     "d'influence."),
    ("Smurfing", "Egalement appele structuring ou fractionnement. Technique "
     "consistant a diviser une somme importante en plusieurs transactions "
     "de montant plus faible pour eviter de declencher les seuils de "
     "declaration automatique."),
    ("Structuring", "Voir Smurfing."),
    ("Compte mule", "Compte bancaire utilise, sciemment ou non par son "
     "titulaire, pour transiter des fonds d'origine frauduleuse avant leur "
     "transfert vers d'autres comptes, souvent a l'etranger."),
    ("Account Takeover (ATO)", "Prise de controle d'un compte client "
     "legitime par un fraudeur, generalement via phishing, vol "
     "d'identifiants ou ingenierie sociale, pour effectuer des operations "
     "non autorisees."),
    ("Phishing", "Technique d'ingenierie sociale visant a obtenir des "
     "informations confidentielles (identifiants, coordonnees bancaires) en "
     "se faisant passer pour un tiers de confiance, generalement par email "
     "ou SMS frauduleux."),
    ("Blanchiment d'argent", "Processus consistant a dissimuler l'origine "
     "illicite de fonds en les faisant transiter par des circuits "
     "financiers legaux, generalement en trois etapes : placement, "
     "empilage (layering), integration."),
    ("Layering (empilage)", "Deuxieme etape du blanchiment d'argent : "
     "multiplication des transactions et des intermediaires pour brouiller "
     "la tracabilite de l'origine des fonds."),
    ("Placement", "Premiere etape du blanchiment d'argent : introduction des "
     "fonds d'origine illicite dans le systeme financier legal."),
    ("Integration", "Derniere etape du blanchiment d'argent : reinjection "
     "des fonds blanchis dans l'economie legale sous une apparence "
     "legitime (achat immobilier, investissement, etc.)."),
    ("Beneficiaire effectif", "Personne physique qui controle in fine une "
     "entite juridique ou pour le compte de qui une transaction est "
     "realisee, meme si elle n'apparait pas nominalement."),
    ("Societe ecran", "Entite juridique creee principalement pour dissimuler "
     "l'identite du beneficiaire effectif ou l'origine de fonds, sans "
     "activite economique reelle."),
    ("SAR / Declaration de soupcon", "Suspicious Activity Report. Signalement "
     "obligatoire adresse a l'autorite competente (TRACFIN en France) "
     "lorsqu'une transaction presente des indices de blanchiment ou de "
     "financement du terrorisme."),
    ("Score de fraude", "Indicateur numerique, generalement de 0 a 100, "
     "produit par un modele de machine learning pour estimer la "
     "probabilite qu'une transaction ou un compte soit frauduleux."),
    ("Faux positif", "Transaction ou compte signale comme suspect par un "
     "systeme de detection alors qu'il ne s'agit pas reellement d'une "
     "fraude. Un taux de faux positifs eleve degrade l'efficacite "
     "operationnelle des equipes d'investigation."),
    ("Faux negatif", "Fraude reelle non detectee par le systeme de "
     "surveillance, generalement decouverte a posteriori ou signalee par "
     "le client."),
    ("Red flag", "Indicateur d'alerte, signal isole ou combine suggerant un "
     "risque de fraude et declenchant une investigation ou un blocage "
     "preventif."),
    ("Gel des avoirs", "Mesure conservatoire empechant tout mouvement sur un "
     "compte, generalement appliquee aux personnes ou entites figurant sur "
     "une liste de sanctions internationales."),
    ("Liste de sanctions", "Liste officielle (ONU, Union Europeenne, OFAC "
     "aux Etats-Unis) de personnes, entites ou pays soumis a des mesures "
     "restrictives, avec lesquels toute relation d'affaires est interdite "
     "ou strictement encadree."),
    ("DSP2", "Directive europeenne sur les Services de Paiement (2eme "
     "version). Impose notamment l'authentification forte du client (SCA) "
     "pour les operations de paiement en ligne."),
    ("Authentification forte (SCA)", "Strong Customer Authentication. "
     "Procedure d'authentification combinant au moins deux facteurs "
     "independants (connaissance, possession, inherence) pour securiser "
     "les operations de paiement."),
    ("DORA", "Digital Operational Resilience Act. Reglement europeen "
     "renforcant les exigences de resilience operationnelle numerique des "
     "etablissements financiers face aux risques cyber."),
    ("Transaction monitoring", "Surveillance automatisee et continue des "
     "transactions d'un client pour detecter des comportements atypiques "
     "par rapport a son profil habituel."),
    ("Profil de risque client", "Evaluation synthetique du niveau de risque "
     "associe a un client, etablie a partir de criteres tels que son "
     "activite, sa localisation, le volume et la nature de ses "
     "transactions."),
    ("Reseau de fraude organisee", "Ensemble de comptes, entites ou "
     "individus lies entre eux et operant de maniere coordonnee pour "
     "commettre des fraudes, generalement identifiable par analyse de "
     "graphe (Graph Neural Network)."),
]

lines = ["GLOSSAIRE — TERMES DE LA DETECTION DE FRAUDE ET DE LA CONFORMITE", "Usage interne", ""]
for terme, definition in GLOSSAIRE_TERMES:
    lines.append(f"{terme.upper()}")
    lines.append(definition)
    lines.append("")

(config.DATA_RAW_DIR / "glossaire_termes_fraude.txt").write_text("\n".join(lines).strip(), encoding="utf-8")
print(f"- glossaire_termes_fraude.txt ({len(GLOSSAIRE_TERMES)} termes)")


# ── 6. Cas clients anonymises (40 fichiers individuels) ───────────────────────

TYPOLOGIES_CAS = [
    "usurpation d'identite",
    "fraude au virement (faux conseiller bancaire)",
    "smurfing / fractionnement de virements",
    "fraude documentaire au credit",
    "prise de controle de compte (account takeover)",
    "fraude interne",
    "fraude a la carte bancaire",
    "blanchiment via compte mule",
]
SEGMENTS_CAS = ["Particulier", "Professionnel", "Banque privee", "PME"]
CANAUX_DETECTION = [
    "alerte automatique du modele XGBoost Transactions",
    "alerte du modele LSTM Comportemental",
    "signalement du GNN Relations (reseau de fraude)",
    "signalement manuel par un conseiller",
    "signalement du client lui-meme",
    "controle de conformite periodique",
]
ISSUES_CAS = [
    "Fraude confirmee, montant integralement recupere apres blocage preventif",
    "Fraude confirmee, perte partielle malgre l'intervention",
    "Fraude non averee apres investigation approfondie, dossier classe",
    "Fraude confirmee, dossier transmis a TRACFIN",
    "Compte cloture a titre conservatoire, plainte deposee par la banque",
    "Fraude confirmee, remboursement partiel negocie avec le client",
]
SIGNAUX_POOL = [
    "transaction superieure a 3 fois la moyenne historique du compte",
    "connexion depuis un pays non habituellement frequente",
    "plusieurs tentatives echouees suivies d'un succes",
    "modification d'adresse suivie d'un virement dans les 24h",
    "virement vers un compte beneficiaire cree il y a moins de 30 jours",
    "transaction a une heure inhabituelle (entre 01h et 05h)",
    "virement fractionne en plusieurs operations similaires",
    "utilisation d'un nouveau dispositif d'authentification",
    "geolocalisation incoherente avec l'adresse de facturation",
    "augmentation soudaine de la frequence de transactions",
]

N_CAS = 40
for i in range(1, N_CAS + 1):
    typologie = random.choice(TYPOLOGIES_CAS)
    segment   = random.choice(SEGMENTS_CAS)
    canal     = random.choice(CANAUX_DETECTION)
    issue     = random.choice(ISSUES_CAS)
    annee     = random.choice([2022, 2023, 2024])
    date_detection = random_date(start_year=annee)
    montant = round(random.choices(
        [random.uniform(800, 6000), random.uniform(6000, 40000), random.uniform(40000, 250000)],
        weights=[55, 35, 10]
    )[0], 2)
    delai_jours = random.randint(1, 21)
    signaux = random.sample(SIGNAUX_POOL, k=random.randint(2, 4))
    client_id = f"CLI-{random.randint(100000, 999999)}"
    pays = random.choice(PAYS)

    signaux_txt = "\n".join(f"- {s.capitalize()}" for s in signaux)

    contenu = f"""CAS-2{i:03d} — Fraude par {typologie}
Segment client : {segment}
Detection : {canal}
Date de detection : {date_detection}
Montant concerne : {montant:,.2f} EUR
Pays de destination des fonds : {pays}

CONTEXTE
Le client ({client_id}) presentait un profil de risque standard sans
antecedent de fraude connu avant cet incident. L'operation a ete initiee
sur un canal habituel du client, ce qui a d'abord limite la detection
immediate par les controles de premier niveau.

CHRONOLOGIE
1. Operation initiale traitee sans blocage automatique (score de fraude
   sous le seuil de blocage immediat au moment de la transaction).
2. Signaux d'alerte remontes par {canal} dans un delai de {delai_jours} jour(s).
3. Qualification de l'alerte par l'equipe d'investigation conformement a la
   procedure standard (voir guide_detection_fraude.txt, etape 1).
4. Decision d'investigation approfondie compte tenu du cumul de signaux.

SIGNAUX DETECTES
{signaux_txt}

ISSUE
{issue}.

Ce cas illustre l'importance du delai de detection : un signalement plus
rapide via {canal} aurait permis de reduire l'exposition financiere sur ce
type de typologie ({typologie}).
"""
    (CAS_DIR / f"cas_{i:04d}.txt").write_text(contenu.strip(), encoding="utf-8")

print(f"- cas/ ({N_CAS} recits de cas anonymises)")

total_docs = 3 + 1 + 1 + N_TRANSACTIONS + N_ALERTES + N_CAS + 1
print(f"\nDonnees generees dans : {config.DATA_RAW_DIR}")
print(f"Volume total estime (documents indexables apres ingestion) : ~{total_docs}")
print("Prochaine etape : python scripts/ingest_data.py")
