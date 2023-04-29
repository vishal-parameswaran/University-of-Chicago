import re
from snorkel.labeling import labeling_function, PandasLFApplier, LabelModel, LFAnalysis
from textblob import TextBlob


#Set up labels for Snorkel. We have created labeling functions corresponding to no risk, low risk, and high risk.

#If an article is not labeled as no risk, low risk, or high risk, Snorkel will Abstain (not choose a label)
ABSTAIN=-1 

#If an article matches with any of the 'no risk' labeling functions (and none of the high/low risk functions),
#it will be labeled as no risk
NR=0

#If an article matches with any of the 'low risk' labeling functions (and none of the high/no risk functions),
#it will be labeled as low risk
LR=1

#If an article matches with any of the 'high risk' labeling functions (and none of the no/low risk functions),
#it will be labeled as high risk
HR=2 #high risk



#Define labeling functions for no risk

#### No Risk Labeling Functions
'''
No risk labeling functions include a variety of key flags in the text that would indicate that the article is either of no risk (but related to a disaster) or is simply an irrelevant article.

Labeling functions included in the 'no risk' bucket include articles discussing:
    - Sports that can be misleading as disasters (Ex. Miami Hurricanes Football Team)
    - Insurance related to disasters (Ex. Filing insurance claim for property damaged by earthquake)
    - Costs or financial aid related to post-disaster (Ex. Fundraisers for Hurricane Matthew victims)
    - Common expressions that utilize words from seed list (Ex. Memes flooded the internet after the game)
    - Locations around the world that are named as disasters (Ex. There is a city in Utah named 'Hurricane')
    - Post-disaster recovery phase (Ex. The town has finally recovered from Hurricane Katrina)
    - Past events (Ex. It's been 10 years since the 2004 indian ocean tsunami)
    - Climate change/research (Ex. Scientists discover that global warming makes hurricanes stronger)
    - Crime (Ex. A recent survivor of Hurricane Matthew was murdered)
''' 

#A lot of articles mention insurance claims in regards to disasters; these should be flagged as no risk
@labeling_function()
def lf_nr_insurance(x):
    return NR if re.search(r"insurance firm\w*|insurance group\w*|insurance compan\w*|insurance agenc\w*|" \
                          "insured loss\w*|uninsured loss\w*|loss\w* insured|loss\w* uninsured|covered insurer\w*|" \
                          "covered insurance|quake insurance|earthquake insurance|quake insurer\w*|earthquake insurer\w*|" \
                          "insurer\w* paid|paid insurance|insurance paid|reinsurance|hurricane insurance|cost insurer\w*|" \
                          "cost insurance|leading insurer\w*|disaster related claim\w*|insurer\w* hit|reinsurer\w*|" \
                          "insurance cover\w*|insurer\w* cover\w*|flood\w* insurance|flood\w* insurer\w*|" \
                          "hurricane insurance|hurricane insurer\w*|tornado insurance|tornado insurer\w*|" \
                          "wildfire insurance|wildfire insurer\w*|tax assess\w*|tax juris\w*|life prison" , 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#A lot of articles (hurricanes specifically) are talking about sports teams; these should be flagged as no risk
@labeling_function()
def lf_nr_sports(x):
    return NR if re.search(r"lady hurricanes sport\w*|miami hurricanes football|hurricanes unranked|carolina hurricanes|" \
                          "hurricanes vs|vs hurricanes|home streak|host\w* hurricanes|hurricanes defeat\w*|hurricanes lost|" \
                          "defeat\w* hurricanes|lost hurricanes|hurricanes won|whl|ahl|western hockey league|" \
                          "university miami hurricanes|hurricanes gym|american hockey league|juniors hurricanes|" \
                          "tulsa golden hurricane|quarterback|semifinals|football|basketball|hockey|lethbridge hurricanes|" \
                          "defeat\w* hurricanes|snap hurricanes|beat hurricanes|hurricanes beat|lost hurricanes|" \
                          "hurricanes won|hurricanes lost|patriots ready battle hurricanes|hurricanes score\w*|" \
                          "hurricane receiver\w*|team champion\w*|nba season|nfl season|nhl season|mbl season|" \
                          "scor\w* ([0-9]|[1-8][0-9]|9[0-9]|1[0-9]{2}|200) points|first half|second half|1st half|" \
                          "2nd half|championship\w*|practice field\w*|baseball|soccer|win game|world series|championship game|" \
                          "stanley cup", 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#A lot of articles talking about costs post-disaster and aplying for aid; these should be flagged as no risk
@labeling_function()
def lf_nr_costs_aid(x):
    return NR if re.search(r"fema(?!\S)|federal emergency management agency|" \
                          "disaster loan\w*|disaster assistance grant\w*|costs hurricane\w*|relief package|" \
                          "recover financially|recover\w* costs|recoup.*cost\w*|\d+ billion\w* damage|" \
                          "\d+ million\w* damage|costl\w* disaster|hurricane related cost\w*|" \
                          "earthquake related cost\w*|wildfire related cost\w*|tsunami related cost\w*|" \
                          "flood\w* related cost\w*|tornado related cost\w*|typhoon related cost\w*|" \
                          "fundrais\w*.*victim\w*|fundrais\w* benefit\w*|hurricane aid\w*|tsunami aid\w*|" \
                          "earthquake aid\w*|tornado aid\w*|wildfire aid\w*|flood\w* aid\w*|disaster grant\w*|" \
                          "emergency grant\w*|cost natural disaster\w*|cost disaster\w*|payment victim\w*|" \
                          "charit\w* organization\w*|charit\w* giving|sentenced \d+ (year\w*|month\w*)|" \
                          "(hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*) donation\w*|" \
                          "donation\w* (hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*)", 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#There is a city in Utah named hurricane. Articles mentioning the town need to be flagged as non-disruptive.
@labeling_function()
def lf_nr_hurricane_utah(x):
    return NR if re.search(r"hurricane utah|hurricane ut(?!\S)|hurricane city utah|hurricane city ut(?!\S)", 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Flooded/flooding is using a lot in language generically (not referring to an actual flood). These are non-disruptive.
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_nr_flood_generic_without_stop(x):
    return NR if re.search(r"sewage leak\w*|faucet leak\w*|sink flood\w*|shower flood\w*|floodwood|" \
                          "toilet flood\w*|flood say(?!\S)|flood says(?!\S)|flood said|flooded countries|" \
                          "flood international market|instagram flooded|facebook flooded|twitter flooded|" \
                          "flood\w* hospital\w*|flood\w* emergency room\w*|flood\w* social media|flood\w* facebook|" \
                          "flood\w* twitter|flood\w* instagram|people flood\w*|floodgate\w*.*open\w*|floodwood|" \
                          "quality crop|basement flood\w*|flood\w* basement|freez\w* pipe\w*|burst\w* pipe\w*|" \
                          "improv\w* flood capacity|hope avoid\w* flood\w*" , 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_nr_flood_generic_with_stop(x):
    return NR if re.search(r"flood of (?!water|rain)|floodgate\w* of(?!\S)|open\w* the floodgate\w*|flood\w* the state with|" \
                            "flood\w* with (?!water|rain)|flood\w* our social wall\w*|flood\w* of idea\w*|" \
                            "flood\w* the scene|flood\w* the net|flood\w* the internet|flood\w* of tear\w*|" \
                            "flooded into health\w*|people flooded|light flooded|refugees flooded|" \
                            "^(?!.*(water|rain\w*)).*flood\w* in\w*|^(?!.*(water|rain\w*)).*flood\w* with",
                            x.first_5_sent, flags=re.I) else ABSTAIN


#The following regex addresses misc mentions of tornado/tsunami that are irrelevant such as sports teams or movies
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_nr_tornado_tsunami_without_stop(x):
    return NR if re.search(r"talladega college|tornado marching band|marching tornado\w*|1966 tornado|" \
                          "golden tornado\w*|tuscaloosa county high school|marching band tornadoes|" \
                          "magical kingdom|emerald city|ruby slippers|judy garland|blue tornado|niger tornadoes|" \
                          "norwich purple tornado\w*|trump tornado|lady tornados|lady tornadoes|tornado ddt|" \
                          "red tornado|tornado aircraft\w*|anoka tornado\w*|anoka high school|scam artist\w*|" \
                          "tornado drill\w*|one foot tsunami|onefoottsunami|tsunami movie|tsunami film|" \
                          "tornado movie|tornado film|tsunami drama|tornado drama|movie review|casting announc\w*|" \
                          "silver tsunami|trillion tsunami|trillion dollar tsunami|stimulus tsunami|" \
                          "intimate tsunami\w*|film critic\w*|movie critic\w*|cinematic tsunami|alzheimers tsunami|" \
                          "alzheimer s tsunami|tsunami owner|tsunami sushi bar|shale tsunami|digital tsunami|" \
                          "pension tsunami|trump tsunami|film director|film producer|defends trump|trump taunts|" \
                          "new tornado prediction system|presidential campaign|online dating|presidential election|" \
                          "next election|boxing day tsunami|2011 tsunami|2004 indian ocean tsunami|" \
                          "indian ocean tsunami", 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#The following regex address misc mentions of tornado that are irrelevant such as sports teams or movies
#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_nr_tornado_tsunami_with_stop(x):
    return NR if re.search(r"tornado of(?!\S)|stepped into a tornado|wizard of oz|tornado\w* of defense|" \
                            "victory over|tsunami of (?!water|rain)|ride that tsunami|like a tsunami|like a hurricane|" \
                            "like an earthquake|likened.*to a tsunami|hailed him as the unstoppable tsunami", 
                            x.first_5_sent, flags=re.I) else ABSTAIN

#The following regex addresses recovery or description of past events
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_nr_recovery_past_without_stop(x):
    return NR if re.search(r"threat pass\w*|warning lift\w*|withdr\w* tsunami warning|withdr\w* hurricane warning|" \
                            "withdr\w* earthquake warning|withdr\w* flood warning|withdr\w* wildfire warning|" \
                            "withdr\w* tornado warning|withdr\w* storm warning|school\w*.*reopen\w*|full recovery|" \
                            "recover\w*.*year\w* (after|later)|continu\w* recover|(recover\w*|relief) effort\w*|" \
                            "(hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*).*cleanup|" \
                            "support relief|rebuild\w* effort\w*|effort\w* rebuild\w*|great white hurricane|" \
                            "(hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*).*aftermath|" \
                            "newspaper\w* apologiz\w*|mak\w* fun|resume\w* operation\w*|resume\w* normal operation\w*", 
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#The following regex addresses recovery or description of past events with stop words
#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_nr_recovery_past_with_stop(x):
    return NR if re.search(r"has been cancel\w*|no reports of.*damage|issued then cancel\w*|issued  then cancel\w*|" \
                            "jobs after wildfire\w*|history of diaster\w*|history of natural disaster\w*|" \
                            "recover\w* from (hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*)|" \
                            "([1-9]|[1-8][0-9]|9[0-9]|[1-8][0-9]{2}|900) (month\w*|year\w*|decade\w*) (since|after|from|follow\w*|later|passed|has passed|have passed|ago)|" \
                            "(a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninety|hundred) (month\w*|year\w*|decade\w*) (since|after|from|follow\w*|later|passed|has passed|have passed|ago)|" \
                            "still recover\w* from|decade\w* ago|successful conclusion|recover\w* work|" \
                            "recover\w* reconstruct\w*|no impact|hero\w* of|lot of smil\w*|brought under control", 
                            x.first_5_sent, flags=re.I) else ABSTAIN

#The following regex addresses climate change without stop words
@labeling_function()
def lf_nr_climate_change(x):
    return NR if re.search(r"climate chang\w*|scientist\w* (discover\w*|research\w*)|(discover\w*|research\w*) scientist|" \
                            "(study|studies) (hurricane\w*|earthquake\w*|flood\w*|wildfire\w*|tornado\w*|tsunami\w*)|" \
                            "global warming|fossil fuel\w*|climate expert\w*|scien\w* world|science week\w*|mystery crack\w*|" \
                            "scien\w* link\w*|carbon emission\w*|discover\w* scientist\w*|study (find\w*|found)|" \
                            "scientist\w* drill\w* deep", 
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#The following regex addresses crime
@labeling_function()
def lf_nr_crime(x):
    return NR if re.search(r"shoot\w* incident\w*|murder\w*|homicid\w*|illegal gun\w*", 
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#The following regex addresses wildfires
@labeling_function()
def lf_nr_wildfire(x):
    return NR if re.search(r"spread like wildfire|wildfire snippet|game of thrones|wildfire prevention|" \
                            "wildfire golf club|like a wildfire by", 
                            x.first_5_sent, flags=re.I) else ABSTAIN


#### Low Risk Labeling Functions
'''
Low risk labeling functions include a variety of key flags in the text that would indicate that the article is indicative of a disruption, but one of lower risk.

Labeling functions included in the 'low risk' bucket include articles discussing:
    - Wildfires that are mostly contained (Ex. The wildfire was 80% contained)
    - Transit that was disrupted (Ex. The tollway was temporary closed due to the severe flooding)
    - Indication of no deaths or minor injuries (Ex. No one was killed from the factory explosion)
    - Information that authitories have situation under control (Ex. There is no immediate threat from the storm)
    - Non-severe weather or low-grade disasters (Ex. The hurricane was classified as category one)
'''

#The regex addresses wildfires that are largely contained
@labeling_function()
def lf_lr_wildfire(x):
    return LR if re.search(r"([78][0-9]|9[0-9]|100) contain\w*|contain\w* ([78][0-9]|9[0-9]|100)|" \
                            "([78][0-9]|9[0-9]|100) percent contain\w*|([78][0-9]|9[0-9]|100) percent control\w*|" \
                            "([78][0-9]|9[0-9]|100) control\w*|control\w* ([78][0-9]|9[0-9]|100)",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function addresses mild transit disruption
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_lr_transit_without_stop(x):
    return LR if re.search(r"operation\w* resum\w*|resum\w* operation\w*|airport\w* reopen\w*|" \
                            "reopen\w* airport\w*|without disrupt\w* air\w*|return\w* normal|" \
                            "remain\w* normal|operat\w* schedul\w*|minim\w* disrupt\w*|" \
                            "disrupt\w* minim\w*|traffic delay\w*|delay\w* traffic|" \
                            "road\w* clos\w*|clos\w* road\w*|clos\w* interstate\w*|" \
                            "interstate\w* clos\w*|clos\w* highway\w*|highway\w* clos\w*|clos\w* street\w*|" \
                            "street\w* clos\w*|freeway\w* clos\w*|clos\w* freeway\w*|road\w* shut\w*|" \
                            "shut\w* road|shut down road|interstate\w* shut\w*|shut\w* interstate|" \
                            "shut down interstate|street\w* shut\w*|shut\w* street|shut down street|" \
                            "freeway\w* shut\w*|shut\w* freeway|shut down freeway\w*|" \
                            "highway\w* shut\w*|shut\w* highway|shut down highway\w*|" \
                            "temporar\w* clos\w*|clos\w* temporar\w*|reopen\w* lane\w*|" \
                            "clos\w* trail\w*|trail\w* clos\w*|clos\w* intersection\w*|" \
                            "intersection\w* clos\w*|clos\w* traffic\traffic clos\w*",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_lr_transit_with_stop(x):
    return LR if re.search(r"some delay\w*|some disrupt\w*|multip\w* disrupt\w*|multip\w* delay\w*",
                            x.first_5_sent, flags=re.I) else ABSTAIN

#This function indicates minor injuries or decreasing death rate as well as minor injuries
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_lr_death_without_stop(x):
    return LR if re.search(r"minor injur\w*|death rate reduc\w*|death toll reduc\w*",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_lr_death_with_stop(x):
    return LR if re.search(r"no one was kill\w*|no report\w* of death\w*|no death\w*|no one kill\w*|" \
                            "no immediate report\w*|no initial report\w*|no injur\w*|no immediate threat",
                            x.first_5_sent, flags=re.I) else ABSTAIN

#This function is a bit broad but represents that authorities have situation under control or in recovery
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_lr_authorities_without_stop(x):
    return LR if re.search(r"situation\w* control\w*|evacuat\w* safe\w*|safe\w* evacuat\w*|under control|" \
                            "monitor\w* situation\w*|situation\w* monitor\w*|lift\w* warn\w*|" \
                            "warn\w* lift\w*|threat\w* cancel\w*|cancel\w* threat\w*|" \
                            "start\w* recover\w*|recover\w* start\w*|recover\w* process|process recover\w*|" \
                            "return\w* home\w*|threat\w* pass\w*|temporar\w* evacuat\w*|evacuat\w* temporar\w*",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_lr_authorities_with_stop(x):
    return LR if re.search(r"only a few report\w*|did not issue|are not issu\w*|didnt issue|" \
                            "stay open|remain open|no threat\w*|no immediate threat\w*|" \
                            "no danger to the public|no danger to public",
                            x.first_5_sent, flags=re.I) else ABSTAIN

#This function flags articles with non-severe weather or low-grade disaster
@labeling_function()
def lf_lr_weather(x):
    return LR if re.search(r"adverse weather alert\w*|storm pass\w*|flash flood\w*|mini tornado\w*|" \
                            "minitornado\w*|category ([1-3])|category three|storm warn\w*|warn\w* storm\w*|" \
                            "heavy rain\w*|light rain\w*|category two|category one|" \
                            "hurricane\w* downgrad\w*|magnitude ([1-5][0-9]|6[0-9])|([1-5][0-9]|6[0-9]) magnitude|" \
                            "([1-5][0-9]|6[0-9]) earthquake|earthquake ([1-5][0-9]|6[0-9])|" \
                            "quake ([1-5][0-9]|6[0-9])|([1-5][0-9]|6[0-9]) quake|" \
                            "magnitude([1-5][0-9]|6[0-9])|([1-5][0-9]|6[0-9])magnitude|" \
                            "([1-5][0-9]|6[0-9])earthquake|earthquake([1-5][0-9]|6[0-9])|" \
                            "quake([1-5][0-9]|6[0-9])|([1-5][0-9]|6[0-9])quake|condition\w* improv\w*|" \
                            "improv\w* condition\w*|subdue\w* fire\w*|fire\w* subdue\w*",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#### High Risk Labeling Functions
'''
High risk labeling functions include a variety of key flags in the text that would indicate that the article is indicative of high risk disruption.

Labeling functions included in the 'high risk' bucket include articles discussing:
    - Emergency warnings (Ex. A state of emergency was declared following the increasing count of Coronavirus cases)
    - Evacution of houses/cities (Ex. The region was evacuated due to the hurricane advisory)
    - Missing people (Ex. There was 27 people reported missing following the tornado)
    - Loss of power (Ex. The storm caused a widespread power outage in New York)
    - Large scale transit disruption (Ex. JFK airport was closed due to the snowstorm)
    - Property Damage (Ex. The tornado destroyed many homes)
    - Health crisis (Ex. The increase in swine flu cases has classified it as a global pandemic)
    - Closures (Ex. The tornado forced the school to shutdown)
    - Man Made Disasters (Ex. The chemical spill caused the highway to close)
    - Death (Ex. 100 people were left dead after the tornado ripped through town)
    - High magnitude earthquakes (Ex. An 8.7 earthquake struck last night)
    - High impact tornados (Ex. An ef-3 tornado ripped through town)
    - High category hurricanes (Ex. A category 4 hurricane is expected to hit in tomorrow in Florida)
'''

#Define labeling functions for high risk

#This is a labeling function for emergency warnings.
@labeling_function()
def lf_hr_emergency(x):
    return HR if re.search(r"state\w* emergenc\w*|emergenc\w* weather warning|global health emergenc\w*|" \
                          "national health emergenc\w*|tornado emergenc\w*|^(?=.*emergenc\w*)(?=.*declar\w*).+|" \
                          "red weather warning|nation\w* emergenc\w*|emergenc\w* assist\w*|severe weather warning\w*|" \
                          "sen\w* militar\w*" , 
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This is a labeling function for evacuation of houses/cities
#Creating a function to be applied on text without stop words here.
@labeling_function()
def lf_hr_evacuate_without_stop(x):
    return HR if re.search(r"advis.*evacuat|displac.*many|^(?=.*order\w*)(?=.*evacuat\w*).+|" \
                          "mand.*evacuat|people.*evacuat|more.*evacuat|unprecedent.*evacuat|displac.*thousand|" \
                          "^(?=.*resident\w*)(?=.*evacuat\w*).+|evacuat.*home|fresh.*evacuat|big\w* evacuat|" \
                          "larg\w* evacuat|resident.*order.*leave|had evacuat|have evacuat|need evacuat|" \
                          "should evacuat|forc.*evacuat|displac.*resident|crew.*evacuat|region.*evacuat|" \
                          "\d+\s+evacuat\w*|\d+\s+people evacuat\w*|\d+\s+resident\w* evacuat\w*|" \
                          "evacuat\w*\s+\d+\s+residen\w*|evacuat\w*\s+\d+\s+person\w*|evacuat\w*\s+\d+\s+people|" \
                          "evacuat\w*\s+\d+\s+home\w*|evacuat\w*\s+\d+\s+house\w*|evacuat\w* around|" \
                          "evacuat\w* approx\w*" , 
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#Same as above function but including stop words to be able to include important regex that would not work without stop words
@labeling_function()
def lf_hr_evacuate_with_stop(x):
    return HR if re.search(r"had evacuat|have evacuat|should evacuat|evacuat\w* around", 
                          x.first_5_sent, flags=re.I) else ABSTAIN

#This labeling function flags people needing to flee home (similar to evacuation)
@labeling_function()
def lf_hr_flee(x):
    return HR if re.search(r"force.*fle|people.*fle|residen.*fle|fle.*high.*ground|" \
                            "fle.*home|fle.*flame|fle.*city|fle.*residen|fle.*tornado|" \
                            "fle.*town",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function flags articles talking about missing people
@labeling_function()
def lf_hr_missing(x):
    return HR if re.search(r"report\w* missing|missing person report\w*|people unaccounted|" \
                          "unaccounted people|person\w* unaccounted|unaccounted person\w*|" \
                          "remain\w* missing|\d+\s+missing|\d+\s+unaccounted",
                           x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function flags loss of power
@labeling_function()
def lf_hr_power(x):
    return HR if re.search(r"power blackout\w*|power outage\w*|power cut|cut power|cut electric\w*|" \
                          "power shut\w*|power out\w*|power los\w*|los\w* power|disrupt.*power|" \
                          "down\w* power line\w*|power.*down|power knock\w*|knock\w* power|" \
                          "outage\w* report\w*|report\w* outage|electric\w* blackout\w*|without power|" \
                          "electric\w* outage\w*|electric\w* cut|electric\w* shut\w*|" \
                          "electric\w* out\w*|electric\w* los\w*|los\w* electric\w*|disrupt.*electric\w*|" \
                          "down\w* elecric\w* line\w*|elecric\w*.*down|elecric\w* knock\w*|knock\w* elecric\w*", 
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function flags larger scale transportation disruption
@labeling_function()
def lf_hr_transit(x):
    return HR if re.search(r"airport\w*.*clos\w*|airport\w*.*disrupt\w*|air travel\w*.*disrupt\w*|" \
                          "^(?=.*airport\w*)(?=.*shutdown).+|shut down.*airport\w*|order\w* ground stop\w*|" \
                          "^(?=.*flight\w*)(?=.*suspen\w*).+|^(?=.*flight\w*)(?=.*cancel\w*).+|" \
                          "^(?=.*flight\w*)(?=.*disrupt\w*).+|^(?=.*airport\w*)(?=.*suspen\w*).+|" \
                          "^(?=.*rail\w*)(?=.*shutdown).+|^(?=.*rail\w*)(?=.*suspen\w*).+|" \
                          "^(?=.*rail\w*)(?=.*disrupt\w*).+|^(?=.*rail\w*)(?=.*cancel\w*).+" ,
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function flags property damage such as homes and buildings
@labeling_function()
def lf_hr_prop_damage(x):
    return HR if re.search(r"^(?=.*destr\w*)(?=.*home\w*).+|^(?=.*destr\w*)(?=.*house\w*).+|" \
                          "^(?=.*destr\w*)(?=.*residen\w*).+|^(?=.*destr\w*)(?=.*structure\w*).+|" \
                          "^(?=.*destr\w*)(?=.*building\w*).+|^(?=.*destr\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*destr\w*)(?=.*propert\w*).+|" \
                          "^(?=.*inundat\w*)(?=.*home\w*).+|^(?=.*inundat\w*)(?=.*house\w*).+|" \
                          "^(?=.*threat\w*)(?=.*home\w*).+|^(?=.*threat\w*)(?=.*house\w*).+|" \
                          "^(?=.*threat\w*)(?=.*building\w*).+|^(?=.*threat\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*threat\w*)(?=.*residen\w*).+|^(?=.*threat\w*)(?=.*structure\w*).+|" \
                          "^(?=.*endanger\w*)(?=.*home\w*).+|^(?=.*endanger\w*)(?=.*house\w*).+|" \
                          "^(?=.*endanger\w*)(?=.*building\w*).+|^(?=.*endanger\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*endanger\w*)(?=.*residen\w*).+|^(?=.*endanger\w*)(?=.*structure\w*).+|" \
                          "^(?=.*consum\w*)(?=.*home\w*).+|^(?=.*consum\w*)(?=.*house\w*).+|" \
                          "^(?=.*consum\w*)(?=.*building\w*).+|^(?=.*consum\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*consum\w*)(?=.*residen\w*).+|^(?=.*consum\w*)(?=.*structure\w*).+|" \
                          "^(?=.*home\w*)(?=.*los\w*).+|^(?=.*house\w*)(?=.*los\w*).+|" \
                          "^(?=.*flam\w*)(?=.*home\w*).+|^(?=.*flam\w*)(?=.*house\w*).+|" \
                          "^(?=.*flam\w*)(?=.*building\w*).+|^(?=.*flam\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*flam\w*)(?=.*residen\w*).+|^(?=.*flam\w*)(?=.*structure\w*).+|" \
                          "^(?=.*devour\w*)(?=.*home\w*).+|^(?=.*devour\w*)(?=.*house\w*).+|" \
                          "^(?=.*devour\w*)(?=.*building\w*).+|^(?=.*devour\w*)(?=.*neighborhood\w*).+|" \
                          "^(?=.*devour\w*)(?=.*residen\w*).+|^(?=.*devour\w*)(?=.*structure\w*).+|" \
                          "^(?=.*tor\w*)(?=.*rooftop\w*).+",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function flags large-scale health crisises such as swine flu, coronavirus, ebola
@labeling_function()
def lf_hr_health(x):
    return HR if re.search(r"^(?=.*ebola)(?=.*outbreak).+|^(?=.*ebola)(?=.*epidemic).+|" \
                          "^(?=.*ebola)(?=.*pandemic).+|regional spread|" \
                          "community spread|^(?=.*measle\w*)(?=.*outbreak).+|" \
                          "^(?=.*measle\w*)(?=.*epidemic).+|^(?=.*measle\w*)(?=.*pandemic).+|" \
                          "^(?=.*coronavirus)(?=.*epidemic).+|^(?=.*coronavirus)(?=.*pandemic).+|" \
                          "^(?=.*coronavirus)(?=.*outbreak).+|^(?=.*ncov2019)(?=.*outbreak).+|" \
                          "^(?=.*ncov2019)(?=.*epidemic).+|^(?=.*ncov2019)(?=.*pandemic).+|" \
                          "^(?=.*ncov19)(?=.*epidemic).+|^(?=.*ncov19)(?=.*pandemic).+|" \
                          "^(?=.*ncov19)(?=.*outbreak).+|^(?=.*2019 ncov)(?=.*outbreak).+|" \
                          "^(?=.*2019 ncov)(?=.*epidemic).+|^(?=.*2019 ncov)(?=.*pandemic).+|" \
                          "^(?=.*19 ncov)(?=.*epidemic).+|^(?=.*19 ncov)(?=.*pandemic).+|" \
                          "^(?=.*19 ncov)(?=.*outbreak).+|^(?=.*2019ncov)(?=.*pandemic).+|" \
                          "^(?=.*2019ncov)(?=.*epidemic).+|^(?=.*2019ncov)(?=.*outbreak).+|" \
                          "^(?=.*19cov)(?=.*epidemic).+|^(?=.*19ncov)(?=.*pandemic).+|" \
                          "^(?=.*19ncov)(?=.*outbreak).+|mandatory quarantin\w*|" \
                          "mandatory quarantin\w*|global spread|mass vaccination|" \
                          "^(?=.*people)(?=.*quarantin\w*).+|^(?=.*passenger)(?=.*quarantin\w*).+|" \
                          "^(?=.*patient)(?=.*quarantin\w*).+|^(?=.*resident)(?=.*quarantin\w*).+|" \
                          "^(?=.*cholera)(?=.*epidemic).+|^(?=.*cholera)(?=.*pandemic).+|" \
                          "^(?=.*cholera)(?=.*outbreak).+|^(?=.*swine flu)(?=.*pandemic).+|" \
                          "^(?=.*swine flu)(?=.*epidemic).+|^(?=.*swine flu)(?=.*outbreak).+|" \
                          "^(?=.*h1n1)(?=.*epidemic).+|^(?=.*h1n1)(?=.*pandemic).+|" \
                          "^(?=.*h1n1)(?=.*outbreak).+|^(?=.*kerala)(?=.*pandemic).+|" \
                          "^(?=.*kerala)(?=.*epidemic).+|^(?=.*kerala)(?=.*outbreak).+|" \
                          "^(?=.*nipah)(?=.*epidemic).+|^(?=.*nipah)(?=.*pandemic).+|" \
                          "^(?=.*nipah)(?=.*outbreak).+|^(?=.*coronavirus)(?=.*pandemic).+|" \
                          "global pandemic|isolation ward\w*|outbreak\w* worsen\w*|viral outbreak\w*|" \
                          "^(?=.*yellow fever)(?=.*outbreak).+|^(?=.*yellow fever)(?=.*pandemic).+|" \
                          "^(?=.*yellow fever)(?=.*epidemic).+|^(?=.*unprecedented)(?=.*outbreak\w*).+",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This is a generif function for natural disasters such as volcano eruptions
@labeling_function()
def lf_hr_natural_disaster(x):
    return HR if re.search(r"^(?=.*volcan\w*)(?=.*erupt\w*).+|^(?=.*earthquak\w*)(?=.*trigger\w*)(?=.*tsunami\w*).+|" \
                          "^(?=.*volatile)(?=.*volcan\w*).+|^(?=.*flood\w*)(?=.*cit\w*).+|disast\w* zone|" \
                          "^(?=.*flam\w*)(?=.*rag\w*).+|battl\w*.*wildfire\w*|^(?=.*large)(?=.*blaze\w*).+|" \
                          "leav\w*.*destruct\w*|^(?=.*catastrophic*)(?=.*wildfire\w*).+|" \
                          "^(?=.*air)(?=.*quality)(?=.*dangerous).+|^(?=.*air)(?=.*quality)(?=.*unsafe).+|" \
                          "^(?=.*huge)(?=.*fire\w*).+|battl\w*.*flam\w*|stay indoors|stay inside|" \
                          "^(?=.*catastrophic*)(?=.*fire\w*).+",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function captures forced closures of schools or businesses
@labeling_function()
def lf_hr_closure(x):
    return HR if re.search(r"^(?=.*school\w*)(?=.*clos\w*).+|^(?=.*shutdown)(?=.*school\w*).+|" \
                          "^(?=.*shutdown)(?=.*business\w*).+",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function captures man-made disasters such as chemical spills or barge collisions
@labeling_function()
def lf_hr_man_made(x):
    return HR if re.search(r"^(?=.*explo\w*)(?=.*chemical\w*)(?=.*plant\w*).+|^(?=.*chemical)(?=.*spill\w*).+|" \
                          "^(?=.*contaminat\w*)(?=.*area\w*).+|barge colli\w*|contaminat\w* water|water contaminat\w*|" \
                          "dump\w* toxic waste|contaminat\w* area|area contaminat\w*|deadly chemical\w*|" \
                          "chemical\w* deadly|hazard\w* chemical\w*|chemical\w* hazard\w*|cancercausing|" \
                          "hazmat respon\w*",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function captured reports of death
@labeling_function()
def lf_hr_death(x):
    return HR if re.search(r"people dead|people died|man died|men died|woman died|women died|danger life|" \
                          "kid\w* dead|kid\w* died|child\w* dead|child\w* died|" \
                          "miner\w* dead|miner\w* died|confirm\w* dea\w*|dea\w* confirm\w*|" \
                          "\d+ dead|\d+ left dead|\d+ people died|\d+ left dead|kill\w* \d+ people|" \
                          "kill\w* \d+ residents|kill\w* \d+ citizens|^(?=.*kill\w*)(?=.*least)(?=.*people).+|" \
                          "people kill\w*|person kill\w*|deadl\w* fire|^(?=.*increas\w*)(?=.*death)(?=.*toll).+|" \
                          "^(?=.*ris\w*)(?=.*death)(?=.*toll).+|kill\w* animal\w*|animal\w* kill\w*|animal\w* dead|" \
                          "dead animal\w*|taken live\w*|live\w* taken|claim\w*.*\d+.*live\w*|" \
                          "\d+ injur\w*|injur\w* \d+|\d+ deaths|burn\w* death|fatalit\w* report\w*|" \
                          "report\w* fatalit\w*|dead\w* tornado\w*|dead\w* earthquake\w*|dead\w* tsunami|" \
                          "tornado dead\w*|earthquake dead\w*|tsunami dead\w*|tsunami kill\w*|kill\w* tsunami|" \
                          "earthquake kill\w*|kill\w* earthquake|tornado kill\w*|kill\w* tornado|" \
                          "wildfire kill\w*|kill\w* wildfire|search rescue|\d+ fatalit\w*|\d+ deaths|" \
                          "kill\w* \d+ fish|many dead|many died",
                          x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This fuction captures high magnitude earthquakes
@labeling_function()
def lf_hr_earthquake(x):
    return HR if re.search(r"earthquake ([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120) earthquake|" \
                            "magnitude ([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120) magnitude|" \
                            "quake ([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120) quake|" \
                            "earthquake([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120)earthquake|" \
                            "magnitude([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120)magnitude|" \
                            "quake([78][0-9]|9[0-9]|1[01][0-9]|120)|([78][0-9]|9[0-9]|1[01][0-9]|120)quake",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN

#This function captures high rating tornados
@labeling_function()
def lf_hr_tornado(x):
    return HR if re.search(r"ef2(?!\S)|ef3(?!\S)|ef4(?!\S)|" \
                            "ef5(?!\S)|f2(?!\S)|f3(?!\S)|f4(?!\S)|f5(?!\S)",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN


#This function captures high category hurricanes
@labeling_function()
def lf_hr_hurricane(x):
    return HR if re.search(r"cat\w* ([3-5])|category three|category four|category five|" \
                            "category 3equivalent|category 4equivalent|category 4equivalent",
                            x.first_5_sent_no_stop, flags=re.I) else ABSTAIN



ABSTAIN=-1
ND = 0
D = 1

#Define very simple labeling functions for disruption

@labeling_function()
def lf_short_text(x):
    return ND if len(x.first_5_sent_no_stop.split()) < 25 else ABSTAIN

@labeling_function()
def lf_seedlist(x):
    return D if re.search(r"quake|hurricane|forest.*fire|tsunami|tornado|cyclone|typhoon|outbreak|wildfire|flood|category 4|category 5|virus|infected|strong.* storm|heavy rain|fog|smog|volcano|epidemic|airport.*disrupt|disrupt.*airport|flight.*cancel|severe.*weather|disrupt.*flight|flight.*delay|chemical.*spill|environment.*hazard|extreme.*weather|mine.*shutdown|mine.*accident|mine.*disaster|factory.*fire|factory.*collapse|building.*collapse|factory.*explode|factory.*explosion|strong.*wind|fire.*area|burn.*area|"\
"fire.*factory|explosion.*factory|explosion.*building", x.first_5_sent_no_stop, flags=re.I) else ABSTAIN


@labeling_function()
def lf_seedlistsentiment(x):
    return D if re.search(r"quake|hurricane|forest.*fire|tsunami|tornado|cyclone|typhoon|outbreak|wildfire|flood|category 4|category 5|virus|infected|strong.* storm|heavy rain|fog|smog|volcano|epidemic|airport.*disrupt|disrupt.*airport|flight.*cancel|severe.*weather|disrupt.*flight|flight.*delay|chemical.*spill|environment.*hazard|extreme.*weather|mine.*shutdown|mine.*accident|mine.*disaster|factory.*fire|factory.*collapse|building.*collapse|factory.*explode|factory.*explosion|strong.*wind|fire.*area|burn.*area|"\
"fire.*factory|explosion.*factory|explosion.*building", x.first_5_sent_no_stop, flags=re.I) and TextBlob(x.first_5_sent_no_stop).sentiment.polarity <0.2  else ABSTAIN

@labeling_function()
def lf_sentiment(x):
    return ND if TextBlob(x.first_5_sent_no_stop).sentiment.polarity >0.2 else ABSTAIN
