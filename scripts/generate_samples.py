"""
scripts/generate_samples.py
Génère des données financières fictives réalistes pour tester le pipeline RAG.

Produit dans data/raw/ :
    - rapport_fraude_2023.txt
    - transactions_suspectes.csv
    - alertes_conformite.json
    - guide_detection_fraude.txt
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


# ── 1. Rapport de fraude narratif ─────────────────────────────────────────────
RAPPORT = """
RAPPORT ANNUEL — DÉTECTION DE FRAUDE ET GESTION DES RISQUES
Exercice 2023 — Direction de la Conformité et du Contrôle Interne

========================================================================
SYNTHÈSE EXÉCUTIVE
========================================================================

Au cours de l'exercice 2023, le département de détection de fraude a traité
un total de 14 872 alertes, dont 1 203 ont donné lieu à une investigation
approfondie. Sur ce périmètre, 287 cas de fraude avérée ont été confirmés,
représentant un préjudice total estimé à 4,7 millions d'euros.

Le taux de fraude confirmée rapporté au volume total de transactions s'établit
à 0,003%, en légère hausse par rapport à 2022 (0,0024%) mais en deçà des
moyennes sectorielles européennes (0,006%).

========================================================================
CHAPITRE 1 — TYPOLOGIES DE FRAUDE IDENTIFIÉES
========================================================================

1.1 FRAUDE PAR USURPATION D'IDENTITÉ (38% des cas)

La fraude par usurpation d'identité représente la menace principale.
Les schémas les plus fréquents observés en 2023 incluent :

- Ouverture frauduleuse de comptes avec des pièces d'identité contrefaites (22%)
- Prise de contrôle de compte (Account Takeover) via phishing (16%)
- Fraude documentaire lors des demandes de crédit (12%)

Montant moyen par dossier : 8 400 €
Délai moyen de détection : 12 jours

1.2 FRAUDE AUX VIREMENTS (29% des cas)

Les fraudes aux virements ont augmenté de 34% par rapport à 2022, portées par
l'essor des arnaques au faux conseiller bancaire.

Patterns détectés :
- Virement vers compte mule avec fractionnemement (smurfing)
- Virement international atypique hors zones habituelles du client
- Modification soudaine du bénéficiaire habituel

Montant moyen : 23 500 €
Délai de détection : 6 jours (amélioration grâce à l'IA temps réel)

1.3 FRAUDE INTERNE (18% des cas)

La fraude interne reste un risque significatif, principalement portée par :
- Détournement de commissions (43% des fraudes internes)
- Modification non autorisée de données clients (31%)
- Accès illicite à des comptes premium (26%)

Secteurs les plus touchés : Gestion patrimoniale, Crédit entreprise

1.4 FRAUDE AUX CHÈQUES ET FAUX BONS DE CAISSE (15% des cas)

En recul structurel du fait de la dématérialisation, mais encore présente
dans les segments clientèle sénior.

========================================================================
CHAPITRE 2 — INDICATEURS D'ALERTE (RED FLAGS)
========================================================================

Les algorithmes de détection s'appuient sur les signaux suivants :

SIGNAUX FORTS (score > 80) :
- Transaction supérieure à 3 fois la moyenne historique du compte
- Connexion depuis un pays non habituellement fréquenté
- Plusieurs tentatives échouées suivies d'un succès
- Modification d'adresse + virement dans les 24h
- Virement vers compte créé il y a moins de 30 jours

SIGNAUX MODÉRÉS (score 50-80) :
- Transaction à une heure inhabituelle (entre 01h et 05h)
- Virement fractionné en plusieurs opérations similaires
- Utilisation d'un nouveau dispositif d'authentification
- Transaction dans un secteur inhabituel (ex: casino pour client sans historique)

SIGNAUX FAIBLES (score 20-50) :
- Augmentation soudaine de la fréquence de transactions
- Géolocalisation incohérente avec l'adresse de facturation
- Utilisation d'un VPN ou IP masquée

========================================================================
CHAPITRE 3 — PERFORMANCE DES MODÈLES DE DÉTECTION
========================================================================

Trois modèles sont en production en 2023 :

MODÈLE A — XGBoost Transactions (déployé Q1 2022)
- Précision : 94,2%
- Rappel : 87,6%
- F1-score : 90,8%
- Faux positifs/jour : 142

MODÈLE B — LSTM Comportemental (déployé Q3 2023)
- Précision : 96,1%
- Rappel : 91,3%
- F1-score : 93,6%
- Faux positifs/jour : 89 (-37% vs Modèle A)

MODÈLE C — Graph Neural Network Relations (pilote Q4 2023)
- En phase de validation sur 5% du flux
- Détection de réseaux de fraude organisée prometteuse
- Résultats préliminaires : +18% de détection des schémas coordonnés

========================================================================
CHAPITRE 4 — RECOMMANDATIONS 2024
========================================================================

1. Déploiement complet du Modèle B sur 100% du flux transactionnel
2. Intégration du GNN en production pour la détection de réseaux
3. Renforcement de l'authentification forte pour les virements > 10 000 €
4. Formation des équipes de back-office sur les nouvelles typologies de fraude
5. Partenariat avec 3 autres établissements pour partage des listes noires

Directeur de la Conformité : Jean-Pierre MARTIN
Date : 15 janvier 2024
""".strip()

(config.DATA_RAW_DIR / "rapport_fraude_2023.txt").write_text(RAPPORT, encoding="utf-8")
print("✓ rapport_fraude_2023.txt")


# ── 2. Transactions suspectes (CSV) ────────────────────────────────────────────
def random_date(start_year=2023):
    start = datetime(start_year, 1, 1)
    return (start + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d %H:%M")

def random_iban():
    return f"FR{random.randint(10,99)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(100,999)}"

PAYS = ["France", "Allemagne", "Espagne", "Italie", "Portugal", "Roumanie", "Nigeria", "Chine", "Ukraine"]
TYPES = ["Virement SEPA", "Virement international", "Paiement carte", "Retrait DAB", "Virement instantané"]
STATUTS = ["ALERTE_HAUTE", "ALERTE_MOYENNE", "ALERTE_FAIBLE", "BLOQUÉE", "EN_INVESTIGATION"]
MOTIFS = [
    "Montant anormalement élevé",
    "Pays de destination inhabituel",
    "Heure de transaction suspecte",
    "Fractionnement détecté (smurfing)",
    "Bénéficiaire jamais utilisé",
    "Vitesse de transactions anormale",
    "Géolocalisation incohérente",
    "Compte bénéficiaire récent (<30j)",
    "Modification d'adresse récente",
]

rows = []
for i in range(150):
    montant = round(random.choices(
        [random.uniform(500, 5000), random.uniform(5000, 50000), random.uniform(50000, 500000)],
        weights=[60, 30, 10]
    )[0], 2)
    rows.append({
        "id_alerte":        f"ALT-2023-{10000 + i}",
        "date_transaction": random_date(),
        "client_id":        f"CLI-{random.randint(100000, 999999)}",
        "type_transaction": random.choice(TYPES),
        "montant_eur":      montant,
        "pays_destination": random.choice(PAYS),
        "iban_beneficiaire": random_iban(),
        "score_fraude":     random.randint(45, 99),
        "statut":           random.choice(STATUTS),
        "motif_alerte":     random.choice(MOTIFS),
        "analyste":         random.choice(["M. Dubois", "S. Chen", "A. Kowalski", "N. Traoré", None]),
        "fraude_confirmee": random.choice([True, False, False, False]),
    })

csv_path = config.DATA_RAW_DIR / "transactions_suspectes.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("✓ transactions_suspectes.csv (150 lignes)")


# ── 3. Alertes conformité (JSON) ──────────────────────────────────────────────
REGLEMENTS = ["LCB-FT", "RGPD", "MIF2", "DSP2", "DORA", "Bâle III"]
NIVEAUX    = ["CRITIQUE", "MAJEUR", "MINEUR", "INFORMATIF"]

alertes = []
for i in range(40):
    alertes.append({
        "id":           f"CONF-{2023000 + i}",
        "date":         random_date(),
        "reglement":    random.choice(REGLEMENTS),
        "niveau":       random.choice(NIVEAUX),
        "description":  random.choice([
            "Dépassement du seuil de déclaration TRACFIN sans signalement",
            "Défaut de vérification KYC pour client à risque élevé",
            "Absence de gel des avoirs pour client listé UE",
            "Transaction suspecte non déclarée dans les délais réglementaires",
            "Utilisation de données personnelles hors consentement",
            "Authentification forte non appliquée pour virement > 30 000 €",
        ]),
        "entite_concernee": random.choice(["Agence Paris 8e", "Direction Crédit", "Service Titres", "Banque Privée"]),
        "action_requise": random.choice([
            "Déclaration immédiate à TRACFIN",
            "Gel préventif du compte",
            "Révision de la procédure KYC",
            "Audit interne dans les 30 jours",
        ]),
        "cloture": random.choice([True, False]),
    })

json_path = config.DATA_RAW_DIR / "alertes_conformite.json"
json_path.write_text(json.dumps(alertes, ensure_ascii=False, indent=2), encoding="utf-8")
print("✓ alertes_conformite.json (40 alertes)")


# ── 4. Guide de détection ─────────────────────────────────────────────────────
GUIDE = """
GUIDE OPÉRATIONNEL — DÉTECTION ET TRAITEMENT DE LA FRAUDE
Version 3.2 — Usage interne

========================================================================
PROCÉDURES D'INVESTIGATION
========================================================================

ÉTAPE 1 : QUALIFICATION DE L'ALERTE

Dès réception d'une alerte système, l'analyste doit :
a) Vérifier le score de fraude : si < 50, classer en surveillance passive
b) Consulter l'historique transactionnel des 90 derniers jours
c) Vérifier si le client figure sur une liste de surveillance (OFAC, UE, ONU)
d) Contacter le client par téléphone si score > 80 et montant > 5 000 €

Délai maximum de qualification : 4 heures ouvrées

ÉTAPE 2 : BLOCAGE PRÉVENTIF

Conditions de blocage immédiat (sans validation client) :
- Score fraude > 95
- Transaction > 100 000 € vers pays à risque GAFI
- Pattern smurfing détecté sur 5 transactions ou plus
- Client identifié sur une liste de sanctions

ÉTAPE 3 : DÉCLARATION TRACFIN

Obligation légale de déclaration si :
- Fraude avérée > 5 000 €
- Soupçon de blanchiment d'argent
- Financement du terrorisme suspecté

Délai légal : 15 jours ouvrés (recommandé : 48h)
Formulaire : TRACFIN-DS (portail sécurisé)

ÉTAPE 4 : CLÔTURE DU DOSSIER

Chaque dossier doit être clos avec :
- Résumé des faits (500 mots minimum)
- Décision motivée (fraude avérée / non avérée / suspendue)
- Montant récupéré ou passé en perte
- Recommandations pour éviter la récurrence

Conservation des dossiers : 10 ans (obligation légale)

========================================================================
CONTACT URGENCES
========================================================================

Cellule fraude 24h/24 : +33 1 XX XX XX XX
Email sécurisé : fraude-urgence@[banque].fr
TRACFIN : https://www.tracfin.gouv.fr
""".strip()

(config.DATA_RAW_DIR / "guide_detection_fraude.txt").write_text(GUIDE, encoding="utf-8")
print("✓ guide_detection_fraude.txt")

print(f"\n✅ Données générées dans : {config.DATA_RAW_DIR}")
print("👉 Prochaine étape : python scripts/ingest_data.py")
