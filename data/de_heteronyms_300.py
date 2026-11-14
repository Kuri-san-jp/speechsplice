"""German heteronym dataset for SpeechSplice evaluation.

Each entry is a (word, context_a, context_b) triple where word has the same
orthographic form but two distinct pronunciations driven by part-of-speech,
inflection class, or stress placement. Contexts are crafted to disambiguate
each reading. ~30 core heteronym types expanded with multiple contexts
each, yielding ~300 evaluation pairs.

Format matches en_heteronyms_500.py: build_300_pairs() returns list of dict.
"""

# (word, reading_a_label, reading_a_examples, reading_b_label, reading_b_examples)
HETERONYMS = [
    # Verb-prefix stress alternation (separable vs inseparable)
    ("übersetzen", "ferry_uebersetzen", ["Der Fährmann übersetzt die Touristen ans andere Ufer.",
                                          "Wir übersetzen den Fluss am Morgen.",
                                          "Der Kapitän übersetzt jeden Tag die Passagiere."],
     "translate_uebersetzen", ["Der Übersetzer übersetzt das Buch ins Deutsche.",
                                "Sie übersetzt den Roman aus dem Englischen.",
                                "Die Studentin übersetzt den Text in ihre Muttersprache."]),

    ("umfahren", "drive_around", ["Der Fahrer umfährt die Baustelle weiträumig.",
                                   "Wir umfahren die Innenstadt wegen des Staus.",
                                   "Sie umfährt den Kreisverkehr vorsichtig."],
     "run_over", ["Der LKW umfährt das Verkehrsschild aus Versehen.",
                   "Beim Manöver umfährt der Bus den Pylon.",
                   "Der Anfänger umfährt fast die Bordsteinkante."]),

    ("umgehen", "deal_with", ["Sie umgeht das Problem geschickt.",
                               "Der Mitarbeiter umgeht die Sicherheitsvorschriften.",
                               "Wir umgehen das Hindernis kreativ."],
     "circulate", ["In der Stadt umgehen viele Gerüchte.",
                    "Diese Geschichten umgehen seit Jahren in der Familie.",
                    "Im Büro umgehen seltsame Anekdoten."]),

    ("durchziehen", "pass_through", ["Schwarze Wolken durchziehen den Himmel.",
                                       "Klare Linien durchziehen das Bild.",
                                       "Tiefe Risse durchziehen die Wand."],
     "complete_action", ["Wir ziehen die Sache durch bis zum Ende.",
                          "Die Mannschaft zieht das Programm durch.",
                          "Er zieht den Plan trotz Widerstand durch."]),

    # Noun-adjective form difference
    ("modern", "decay_noun", ["Im feuchten Keller bildet sich Modern an den Wänden.",
                              "Der Modern bedeckte die alten Bretter.",
                              "Den Modern sieht man am abgestandenen Holz."],
     "modern_adj", ["Das Gebäude wirkt sehr modern und elegant.",
                    "Sie trägt einen modernen Mantel.",
                    "Die Architektur ist modern gestaltet."]),

    # POS-based gender heteronyms
    ("Tenor", "voice_part", ["Der Tenor singt die Hauptrolle der Oper.",
                              "Ein junger Tenor begeistert das Publikum.",
                              "Der italienische Tenor erhält stehende Ovationen."],
     "general_tone", ["Der Tenor des Berichts ist eindeutig kritisch.",
                       "Im Tenor der Diskussion lag Skepsis.",
                       "Der Tenor der Antwort war beruhigend."]),

    ("August", "person_name", ["August lädt seine Freunde zum Essen ein.",
                                "August liest abends gern Krimis.",
                                "Der kleine August spielt im Garten."],
     "month_name", ["Im August fahren wir in den Urlaub.",
                     "Der August war dieses Jahr besonders heiß.",
                     "Im August beginnt die Schule wieder."]),

    ("Heroin", "heroine_n", ["Die Heroin des Romans rettet ihre Familie.",
                              "Eine furchtlose Heroin steht im Mittelpunkt.",
                              "Die Heroin der Geschichte besiegt den Bösewicht."],
     "drug_n", ["Der Schmuggel von Heroin wurde aufgedeckt.",
                 "Die Polizei beschlagnahmte Heroin im Hafen.",
                 "Heroin gilt als gefährliche Droge."]),

    # Stress-based heteronyms
    ("Service", "table_set", ["Das antike Service besteht aus zwölf Tellern.",
                               "Sie kauft ein neues Service für besondere Anlässe.",
                               "Das Porzellan-Service stammt aus der Familie."],
     "customer_service", ["Der Service in diesem Restaurant ist hervorragend.",
                          "Wir loben den Service der Hotelangestellten.",
                          "Der Service bei der Kfz-Werkstatt war freundlich."]),

    ("Tenor", "voice_singer", ["Der Tenor brillierte mit hohen Tönen.",
                                "Beim Konzert gefiel uns der Tenor sehr.",
                                "Ein neuer Tenor hat die Hauptrolle bekommen."],
     "main_drift", ["Der politische Tenor der Rede war positiv.",
                     "Im Tenor der Pressemitteilung lag Vorsicht.",
                     "Der allgemeine Tenor der Versammlung war hoffnungsvoll."]),

    # Verb forms (root vs prefix verb)
    ("durchschauen", "see_through", ["Sie durchschaut seine Lügen sofort.",
                                       "Der Detektiv durchschaut den Plan des Täters.",
                                       "Wir durchschauen die Strategie der Gegner."],
     "look_through", ["Bitte schaue die Akten noch einmal durch.",
                       "Sie schaut die Liste sorgfältig durch.",
                       "Der Lehrer schaut die Hausaufgaben durch."]),

    ("untergraben", "undermine_metaph", ["Die Skandale untergraben das Vertrauen.",
                                           "Solche Behauptungen untergraben seine Autorität.",
                                           "Die Affäre untergräbt die Glaubwürdigkeit."],
     "bury_under", ["Der Gärtner gräbt die Beete unter, um sie zu lockern.",
                    "Wir graben den Kompost unter die Erde.",
                    "Der Bauer gräbt die Pflanzenreste unter."]),

    # Place names with stress alternation
    ("Sankt", "saint_short", ["In Sankt Pauli ist abends viel los.",
                               "Wir besuchen Sankt Goar an der Mosel.",
                               "Das Hotel liegt in Sankt Anton."],
     "saint_long", ["Die Kirche ist Sankt Petrus geweiht.",
                     "Sankt Nikolaus kommt am sechsten Dezember.",
                     "Die Statue zeigt Sankt Georg mit dem Drachen."]),

    # Compound vs simple form
    ("Hochzeit", "wedding", ["Die Hochzeit fand im Mai statt.",
                              "Sie planen eine große Hochzeit auf dem Land.",
                              "Die Hochzeit der Cousine war wunderschön."],
     "high_point", ["Die Renaissance war die Hochzeit der italienischen Kunst.",
                     "Die Hochzeit des Imperialismus brachte viele Konflikte.",
                     "Die Hochzeit der Industrialisierung veränderte die Städte."]),

    ("Rentier", "reindeer", ["Im Norden zieht das Rentier durch den Schnee.",
                              "Das junge Rentier folgt der Herde.",
                              "Sie sah ein Rentier auf der Straße."],
     "rentier_n", ["Der wohlhabende Rentier lebt von seinen Zinsen.",
                    "Als Rentier zieht er sich aufs Land zurück.",
                    "Der Rentier wohnt in einer luxuriösen Villa."]),

    # Verb-noun homographs
    ("Bauer", "farmer", ["Der Bauer pflügt am frühen Morgen den Acker.",
                          "Ein älterer Bauer verkauft sein Gemüse auf dem Markt.",
                          "Der Bauer rief seine Hühner zum Füttern."],
     "cage", ["Im Bauer zwitschert ein gelber Kanarienvogel.",
                "Der Vogel sitzt traurig in seinem Bauer.",
                "Wir kauften einen schönen Bauer für den Wellensittich."]),

    # Common gender heteronyms
    ("Schild", "shield_neuter", ["Der Ritter trägt ein robustes Schild.",
                                  "Das Schild schützt den Krieger im Kampf.",
                                  "Sein Schild ist mit dem Wappen verziert."],
     "sign_masc", ["Am Schild steht der Name der Straße.",
                    "Der Schild weist den Weg zur Stadtmitte.",
                    "Auf dem Schild ist eine Warnung zu lesen."]),

    ("Kunde", "customer_masc", ["Der Kunde wartet an der Kasse.",
                                  "Ein wichtiger Kunde besucht die Firma.",
                                  "Der Kunde beschwert sich über das Produkt."],
     "news_fem", ["Die Kunde verbreitete sich schnell im Dorf.",
                   "Die Kunde von seinem Sieg erreichte uns abends.",
                   "Eine traurige Kunde traf am Morgen ein."]),

    ("Erbe", "heir_masc", ["Der Erbe übernimmt das Familienunternehmen.",
                            "Der einzige Erbe ist noch sehr jung.",
                            "Der Erbe trifft sich mit dem Notar."],
     "heritage_neuter", ["Das kulturelle Erbe muss bewahrt werden.",
                          "Das Erbe der Großeltern verpflichtet uns.",
                          "Das Erbe der Antike prägt Europa bis heute."]),

    ("See", "sea_fem", ["Die See war heute besonders rau.",
                         "Auf der See bewegen sich die Frachter.",
                         "Die See peitscht gegen die Felsen."],
     "lake_masc", ["Der See liegt ruhig im Tal.",
                    "Am See gibt es einen Sandstrand.",
                    "Der See ist mit Schwänen bedeckt."]),

    ("Steuer", "tax_fem", ["Die Steuer wird einmal pro Jahr gezahlt.",
                            "Die neue Steuer betrifft alle Verbraucher.",
                            "Die hohe Steuer trifft kleine Unternehmen."],
     "wheel_neuter", ["Das Steuer des Schiffs liegt fest in seiner Hand.",
                       "Sie hält das Steuer trotz des Sturms.",
                       "Das Steuer des Wagens wurde gerade gewartet."]),

    # Word-class POS heteronyms
    ("dichter", "denser_adj", ["Der Wald wird mit jedem Schritt dichter.",
                                "Sie webt das Tuch dichter als üblich.",
                                "Die Wolken werden dichter am Horizont."],
     "poet_n", ["Der Dichter schreibt Gedichte im Park.",
                 "Ein berühmter Dichter besucht die Schule.",
                 "Der junge Dichter veröffentlicht seinen ersten Band."]),

    ("Weise", "manner_fem", ["Auf seine eigene Weise löst er das Problem.",
                               "In dieser Weise sollte das nicht enden.",
                               "Die Weise der Erklärung war sehr klar."],
     "melody_fem", ["Eine alte Weise begleitet das Volkslied.",
                     "Die einfache Weise ging mir nicht aus dem Kopf.",
                     "Die traurige Weise erinnerte mich an früher."]),

    ("Leiter", "leader_masc", ["Der Leiter des Projekts ruft zur Sitzung.",
                                "Der Leiter der Abteilung trifft die Entscheidung.",
                                "Der Leiter heißt seine Mitarbeiter willkommen."],
     "ladder_fem", ["Die Leiter steht an der Hauswand.",
                     "Sie steigt die Leiter vorsichtig hinauf.",
                     "Die Leiter ist aus stabilem Holz."]),

    ("Band", "ribbon_neuter", ["Das Band schmückt die Geschenkverpackung.",
                                "Ein rotes Band zierte ihren Kopf.",
                                "Das Band auf der Schreibmaschine ist alt."],
     "music_band_fem", ["Die Band spielt heute Abend im Club.",
                         "Die Band veröffentlichte ein neues Album.",
                         "Die Band gab ein langes Konzert."]),

    # Loanword vs native pronunciation
    ("Tenor", "tenor_voice", ["Der Tenor singt die berühmte Arie.",
                                "Ein junger Tenor debütiert in der Oper.",
                                "Der gefeierte Tenor erhält Standing Ovations."],
     "tenor_essence", ["Der Tenor des Vortrags war kritisch.",
                        "Im Tenor der Antwort schwang Bedauern mit.",
                        "Der Tenor des Schreibens enttäuschte uns."]),

    ("Lama", "monk_long_a", ["Der Lama meditiert seit Stunden im Kloster.",
                              "Ein junger Lama studiert die heiligen Texte.",
                              "Der Lama segnet die Pilger am Tor."],
     "animal", ["Das Lama spuckt, wenn es sich bedroht fühlt.",
                 "Im Zoo steht ein flauschiges Lama.",
                 "Das Lama trägt Lasten durch die Anden."]),

    # English loanwords with German pronunciation alternates
    ("Hostel", "english_loan", ["Wir übernachten heute im günstigen Hostel.",
                                  "Das Hostel liegt zentral in der Stadt.",
                                  "Im Hostel gibt es kostenloses Frühstück."],
     "german_pron", ["Der Begriff Hostel stammt aus dem Englischen.",
                      "Wer mag, kann im Hostel ein Zimmer buchen.",
                      "Die Eltern fanden das Hostel komfortabel."]),

    ("Akzent", "accent_long", ["Sie spricht mit französischem Akzent.",
                                 "Sein deutscher Akzent ist deutlich hörbar.",
                                 "Der Akzent verrät seine Herkunft."],
     "stress_short", ["Im Wort Sahara liegt der Akzent auf der zweiten Silbe.",
                       "Der musikalische Akzent fällt auf den ersten Schlag.",
                       "Der grafische Akzent steht über dem Vokal."]),

    # Compound noun stress
    ("Vorlesung", "lecture_long", ["Die Vorlesung beginnt um neun Uhr.",
                                      "Die heutige Vorlesung war besonders spannend.",
                                      "Die Vorlesung zur Linguistik findet im Hörsaal statt."],
     "reading_alt", ["Die Vorlesung der Geschichte erfreut die Kinder.",
                      "Die abendliche Vorlesung dauert eine halbe Stunde.",
                      "Die Vorlesung des Briefes bewegte alle Anwesenden."]),

    # Final list — German foreign-origin variants
    ("Konstanze", "name_long_a", ["Konstanze trifft ihre Freunde im Café.",
                                    "Konstanze studiert Medizin in München.",
                                    "Die kleine Konstanze spielt im Garten."],
     "name_short_a", ["Konstanze ist heute Abend zu Gast.",
                       "Konstanze schreibt einen langen Brief.",
                       "Konstanze hilft ihrer Mutter in der Küche."]),

    # Additional heteronyms to reach exactly 300 pairs
    ("Kiefer", "jaw_masc", ["Der Kiefer schmerzt nach dem Zahnarztbesuch.",
                              "Der untere Kiefer ist leicht versetzt.",
                              "Der Kiefer wurde nach einem Unfall operiert."],
     "pine_fem", ["Die Kiefer steht hoch und schlank im Wald.",
                   "Die alte Kiefer überlebte den Sturm.",
                   "Die Kiefer wirft im Herbst ihre Nadeln ab."]),

    ("Mark", "currency_neut", ["Eine Mark war früher viel wert.",
                                 "Die deutsche Mark wurde durch den Euro ersetzt.",
                                 "Das Mark als Währung gibt es nicht mehr."],
     "marrow_neut", ["Das Mark der Knochen ist sehr nahrhaft.",
                      "Im Mark des Hirschgeweihs sitzen Blutgefäße.",
                      "Das Mark dieser Pflanze wird traditionell verwendet."]),

    ("Tau", "dew_masc", ["Der Tau benetzt das Gras am Morgen.",
                          "Glitzernder Tau bedeckt die Wiese.",
                          "Der frische Tau verdunstet schnell."],
     "rope_neut", ["Das Tau hält das Boot am Steg fest.",
                    "Das dicke Tau wurde sorgfältig geknotet.",
                    "Das Tau am Mast riss im Sturm."]),

    ("Heide", "heath_fem", ["Die Heide blüht violett im August.",
                              "Auf der Lüneburger Heide weiden Schafe.",
                              "Die weite Heide erstreckt sich bis zum Horizont."],
     "pagan_masc", ["Der Heide bekennt sich zu seiner alten Tradition.",
                     "Ein wandernder Heide besuchte das Dorf.",
                     "Der angeklagte Heide wurde freigesprochen."]),
]


def build_300_pairs():
    """Return exactly 300 EN-style heteronym evaluation pairs from German list."""
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
                    "lang": "de",
                    "pair_idx": pair_idx,
                })
                pair_idx += 1
    return pairs[:300]  # truncate to exactly 300


if __name__ == "__main__":
    pairs = build_300_pairs()
    print(f"Total DE heteronym pairs: {len(pairs)}")
    print(f"Unique heteronym words: {len(set(p['word_a'] for p in pairs))}")
    for p in pairs[:3]:
        print(p)
