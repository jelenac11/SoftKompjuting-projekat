# SoftKompjuting-projekat

Projekat iz predmeta Soft-kompjuting <br/>
Članovi tima: Jelena Cupać sw13/2017, Aleksa Goljović sw14/2017
# Customer Support Chatbot <br/>
Cilj projekta je davanje smislenih odgovora na korisnikove poruke u realnom vremenu. Korisnik ne treba da bude svestan da razgovara sa chatbot-om već sa pravom osobom.
Skup podataka je preuzet sa <a href="https://www.kaggle.com/thoughtvector/customer-support-on-twitter">linka</a>. Ovaj skup podataka broji preko 3 miliona tvitova i odgovara na iste u vezi sa korisničkom podrškom najvećih brendova (Apple, Amazon, Uber...).
# Requirements
Pre pokretanja je potrebno instalirati sledeće biblioteke:
• seaborn <br/>
• tensorflow <br/>
• pandas <br/>
• keras <br/>
• tensorlayer <br/>
• numpy <br/>
• emot <br/>
• spellchecker <br/>
• pyspellchecker <br/>

# Pokretanje
Treniranje mreže je potrebno izvršiti u Jypiter NoteBook-u. <br/>
Importovati chatbot.ipynb u Notebook i redom pokretati ćelije. <br/>
Nakon treniranja, istrenirana mreža se može testirati u Flask aplikaciji. <br/>
Potrebno je istrenirane modele prekopirati u Flask aplikaciju, zatim run-ovati main i otvoriti browser na localhost:5000.

