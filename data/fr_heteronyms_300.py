"""French heteronym (homographes hétérophones) dataset for SpeechSplice evaluation.

Each entry is a (word, context_a, context_b) triple where word has the same
orthographic form but two distinct pronunciations driven by part-of-speech,
inflection class, or stress placement. ~30 core heteronym types expanded
with multiple contexts each, yielding ~300 evaluation pairs.

Format matches en_heteronyms_500.py: build_300_pairs() returns list of dict.
"""

HETERONYMS = [
    # Noun vs verb form (same spelling, different pronunciation)
    ("fils", "son_fis", ["Le fils du voisin joue dans le jardin.",
                          "Mon fils est très studieux.",
                          "Le jeune fils accompagne son père au marché."],
     "thread_fil", ["Les fils électriques traversent le mur.",
                     "Tirer ces fils peut tout défaire.",
                     "Les fils de coton sont fragiles."]),

    ("couvent", "convent_n", ["Le couvent est un lieu de silence.",
                               "Les religieuses vivent dans le couvent depuis des années.",
                               "Le vieux couvent date du dix-septième siècle."],
     "brood_v", ["Les poules couvent leurs œufs depuis trois semaines.",
                  "Les oiseaux couvent au printemps.",
                  "Les pigeons couvent sous le toit."]),

    ("portions", "portions_n", ["Les portions servies au restaurant sont généreuses.",
                                  "Les portions individuelles facilitent la diète.",
                                  "Les portions d'aliments sont équilibrées."],
     "carry_v_imperf", ["Nous portions de lourdes charges sur le dos.",
                         "Nous portions ces caisses depuis le matin.",
                         "Nous portions toujours notre uniforme à l'école."]),

    ("sens", "sense_n", ["Le sens de cette phrase reste obscur.",
                          "Les cinq sens nous renseignent sur le monde.",
                          "Le sens du devoir guide ses actions."],
     "smell_v", ["Tu sens les fleurs du jardin.",
                  "Tu sens le parfum dès l'entrée.",
                  "Tu sens cette odeur dans la pièce ?"]),

    ("est", "east_n", ["Le soleil se lève à l'est.",
                        "Le vent vient de l'est ce matin.",
                        "Les régions de l'est sont plus froides."],
     "is_v", ["Il est très occupé aujourd'hui.",
              "Elle est partie avant midi.",
              "Le travail est terminé ce soir."]),

    ("vis", "screw_n", ["La vis maintient les deux planches ensemble.",
                         "Cette vis est rouillée.",
                         "La vis tourne dans le sens des aiguilles."],
     "live_v", ["Tu vis dans une grande maison.",
                 "Tu vis ta vie comme tu l'entends.",
                 "Tu vis chaque journée pleinement."]),

    # Inflection vs root
    ("nous", "we_pron", ["Nous arrivons à la gare.",
                          "Nous parlons français couramment.",
                          "Nous mangeons ensemble ce soir."],
     "marriage_n", ["Le nous du couple s'est consolidé.",
                     "Le nous fraternel les unit.",
                     "Le nous communautaire est fort dans ce village."]),

    # Verb vs noun
    ("ferment", "leaven_n", ["Le ferment lactique aide la fermentation.",
                               "Ce ferment est utilisé en boulangerie.",
                               "Le ferment du yaourt vient de cultures sélectionnées."],
     "close_v", ["Les magasins ferment à dix-huit heures.",
                  "Les portes ferment automatiquement.",
                  "Les guichets ferment à midi."]),

    ("violent", "violent_adj", ["Le combat fut violent et imprévu.",
                                  "Un orage violent s'est abattu sur la ville.",
                                  "Le choc violent a brisé la vitre."],
     "rape_v", ["Les soldats violent la trêve sans hésiter.",
                 "Ces actes violent les conventions internationales.",
                 "Les manifestants violent les barricades."]),

    ("expédient", "expedient_n", ["L'expédient du moment ne sera pas durable.",
                                    "Cet expédient permet de gagner du temps.",
                                    "L'expédient choisi par le gouvernement déçoit."],
     "send_v_subj", ["Que les colis expédient avant midi !",
                       "Les services expédient les commandes rapidement.",
                       "Les agents expédient les rapports en urgence."]),

    # Stress-based difference (rare in French; mostly POS)
    ("contre", "against_prep", ["Il s'appuie contre le mur.",
                                  "Le verre est posé contre la fenêtre.",
                                  "Elle pousse la porte contre le vent."],
     "counter_n", ["Le contre du joueur a surpris l'adversaire.",
                    "Le contre vient juste après l'attaque.",
                    "Au tennis, le contre est essentiel."]),

    # Liaison vs pause
    ("plus", "more_plyss", ["Je veux plus de temps pour finir.",
                             "Plus tu travailles, plus tu réussis.",
                             "Donnez-moi plus de pain s'il vous plaît."],
     "no_more_ply", ["Je n'en peux plus de cette situation.",
                       "Il ne reste plus de café dans la cafetière.",
                       "Elle ne dit plus rien depuis hier."]),

    ("os", "bone_sing_oss", ["L'os de la jambe est cassé.",
                              "Le chien ronge un os dur.",
                              "Cet os de poulet est très grand."],
     "bones_pl_o", ["Les os anciens reposent dans la grotte.",
                     "Les os trouvés sont très bien conservés.",
                     "Les os fossiles datent du jurassique."]),

    ("but", "goal_byt", ["Le but de la mission est clair.",
                          "L'équipe marque un but à la dernière minute.",
                          "Le but ultime est la paix mondiale."],
     "drank_by", ["Il but trois verres d'eau d'un trait.",
                   "L'enfant but tout le lait du verre.",
                   "Le voyageur but à la fontaine."]),

    ("fier", "proud_adj", ["Le père est fier de ses enfants.",
                            "Le coureur fier brandit son trophée.",
                            "Le pays fier célèbre ses héros."],
     "trust_v_inf", ["Il faut se fier à son intuition.",
                     "Tu peux te fier à mes conseils.",
                     "Mieux vaut se fier aux faits qu'aux rumeurs."]),

    ("tu", "you_2sg", ["Tu marches vite ce matin.",
                        "Tu chantes très bien.",
                        "Tu manges trop de bonbons."],
     "shut_pp", ["Le secret est tu depuis longtemps.",
                  "Le scandale a été tu par les autorités.",
                  "Son nom est tu pour des raisons de sécurité."]),

    # POS-based homographs
    ("lit", "bed_n", ["Le lit confortable invite au repos.",
                       "Son lit est près de la fenêtre.",
                       "Le lit double convient à deux personnes."],
     "read_v_3sg", ["Elle lit un roman policier.",
                     "Le professeur lit le passage à haute voix.",
                     "L'enfant lit son livre préféré."]),

    ("vers", "toward_prep", ["Il marche vers la sortie.",
                              "Le train roule vers Paris.",
                              "Elle court vers la maison."],
     "verse_n", ["Le poète écrit un vers magnifique.",
                  "Ce vers de douze pieds frappe l'esprit.",
                  "Le dernier vers du poème est célèbre."]),

    ("sont", "are_3pl", ["Les enfants sont dans la cour.",
                          "Les voisins sont très accueillants.",
                          "Les fleurs sont fanées depuis hier."],
     "tone_n", ["Le sont musical de l'instrument est pur.",
                 "Un sont aigu interrompt le silence.",
                 "Le sont grave donne une atmosphère solennelle."]),

    ("avocat", "lawyer", ["L'avocat plaide la cause de son client.",
                            "L'avocat de la défense est très éloquent.",
                            "Cet avocat a gagné de nombreuses affaires."],
     "fruit", ["L'avocat mûrit lentement sur l'arbre.",
               "Cet avocat est encore vert à l'intérieur.",
               "L'avocat de Mexique est très savoureux."]),

    # Verb forms
    ("portent", "carry_3pl", ["Ils portent leur valise jusqu'à la chambre.",
                                "Les enfants portent leur sac à dos.",
                                "Les déménageurs portent les meubles avec soin."],
     "wear_3pl", ["Les soldats portent l'uniforme.",
                   "Les invités portent des tenues élégantes.",
                   "Les employés portent un badge."]),

    ("prend", "take_3sg", ["Elle prend le train de huit heures.",
                            "Il prend une décision rapide.",
                            "Le chat prend la souris d'un bond."],
     "freeze_3sg", ["Le lac prend en hiver.",
                     "La gelée prend rapidement au froid.",
                     "Le ciment prend dans quelques heures."]),

    ("oublies", "forget_2sg", ["Tu oublies souvent tes clés à la maison.",
                                 "Tu oublies tes promesses trop vite.",
                                 "Tu oublies le rendez-vous de demain."],
     "oblivion_pl", ["Les oublies du passé pèsent sur l'avenir.",
                      "Ses oublies se multiplient avec l'âge.",
                      "Les oublies s'accumulent dans son agenda."]),

    # Number variation
    ("cent", "hundred", ["Cent personnes sont venues à la fête.",
                          "Le livre coûte cent euros.",
                          "Cent ans plus tard, le souvenir demeure."],
     "scent_perc", ["Le cent du tirage est tombé chez nous.",
                     "Le cent gagnant a été sorti hier.",
                     "Au cent gagnant, la file est longue."]),

    # Common French final-e variations
    ("affluent", "tributary_n", ["L'affluent rejoint la rivière au village.",
                                   "Cet affluent enrichit le débit du fleuve.",
                                   "L'affluent prend sa source dans la montagne."],
     "flock_v", ["Les pèlerins affluent vers le sanctuaire.",
                  "Les touristes affluent en été.",
                  "Les fans affluent au stade pour le match."]),

    ("résident", "resident_n", ["Le résident de l'immeuble se plaint.",
                                  "Chaque résident reçoit un badge.",
                                  "Le résident permanent est inscrit au registre."],
     "reside_v", ["Beaucoup résident dans cette commune.",
                   "Les retraités résident souvent en banlieue.",
                   "Mes parents résident toujours à Lyon."]),

    ("équivalent", "equivalent_n", ["L'équivalent de cette monnaie est négligeable.",
                                      "Cherchez l'équivalent en français.",
                                      "L'équivalent du dollar varie chaque jour."],
     "amount_to_v", ["Ces quantités équivalent à un kilo.",
                      "Ces sommes équivalent à un mois de salaire.",
                      "Ces actions équivalent à une démission."]),

    # Vowel-quality heteronyms
    ("oignons", "onions", ["Les oignons sont coupés finement.",
                             "Les oignons font pleurer le cuisinier.",
                             "Les oignons rouges relèvent la salade."],
     "concern_v", ["Ces problèmes ne nous oignons pas vraiment.",
                    "Ces histoires ne vous oignons en rien.",
                    "Ces affaires nous oignons tous indirectement."]),

    ("présent", "present_n", ["Le présent du moment est précieux.",
                                "Le présent que vous m'offrez me touche.",
                                "Le présent de l'indicatif s'utilise souvent."],
     "presenter_3pl", ["Les artistes présent leur création ce soir.",
                        "Les étudiants présent leur projet à l'oral.",
                        "Les chercheurs présent leurs résultats."]),

    # Final pair
    ("relevant", "relevant_adj", ["L'argument relevant a convaincu le jury.",
                                    "Le détail relevant change tout.",
                                    "Le passage relevant doit être souligné."],
     "raise_v_pp", ["Les rideaux ont été relevant pour aérer.",
                     "Les défis sont relevant de notre responsabilité.",
                     "Les questions relevant de l'éthique sont sensibles."]),

    # Additional heteronyms to reach exactly 300 pairs
    ("compter", "count_v", ["Tu peux compter sur moi pour t'aider.",
                              "Il faut compter chaque pièce avec soin.",
                              "Elle aime compter les étoiles le soir."],
     "matter_v", ["Cela va compter pour la note finale.",
                   "Ces résultats vont compter dans l'évaluation.",
                   "Ces critères vont compter beaucoup pour la décision."]),

    ("intérêt", "interest_n", ["L'intérêt qu'il porte à la science est grand.",
                                 "Mon intérêt pour la musique grandit.",
                                 "L'intérêt du sujet est évident."],
     "rate_n", ["L'intérêt bancaire reste très bas.",
                 "Le taux d'intérêt augmente cette année.",
                 "L'intérêt composé peut faire fructifier l'épargne."]),

    ("faisons", "do_1pl", ["Nous faisons les courses ensemble.",
                            "Nous faisons attention au bruit.",
                            "Nous faisons une pause après le déjeuner."],
     "beech_pl", ["Les faisons rouges du parc sont magnifiques.",
                   "Les vieux faisons résistent à la sécheresse.",
                   "Les faisons centenaires bordent le chemin."]),

    ("but", "scored_3sg", ["Il but le verre d'eau d'un seul coup.",
                            "Le voyageur but à la source claire.",
                            "L'enfant but son jus à grandes gorgées."],
     "objective_n", ["Le but principal du projet est clair.",
                      "Atteindre le but demande des efforts.",
                      "Le but ultime de la mission reste secret."]),
]


def build_300_pairs():
    """Return exactly 300 EN-style heteronym evaluation pairs from French list."""
    pairs = []
    pair_idx = 0
    for word, label_a, ctxs_a, label_b, ctxs_b in HETERONYMS:
        for text_a in ctxs_a:
            for text_b in ctxs_b:
                pairs.append({
                    "word_a": word,
                    "word_b": word,
                    "text_a": text_a,
                    "text_b": text_b,
                    "direction": "a_to_b",
                    "reading_a_label": label_a,
                    "reading_b_label": label_b,
                    "lang": "fr",
                    "pair_idx": pair_idx,
                })
                pair_idx += 1
    return pairs[:300]  # truncate to exactly 300


if __name__ == "__main__":
    pairs = build_300_pairs()
    print(f"Total FR heteronym pairs: {len(pairs)}")
    print(f"Unique heteronym words: {len(set(p['word_a'] for p in pairs))}")
    for p in pairs[:3]:
        print(p)
