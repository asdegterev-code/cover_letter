Das Programm erstellt ein Beispiel eines Initiativbriefes/Anschreibens auf Basis von
* Ihrem Lebenslauf, einschließlich Zeugnisse, Zertifikate und Diplome (Datei Lebenslauf-etc.md)
* Regeln für Initiativbriefe/Anschreiben (Datei Regeln.md) 
* Firmen/Stellenbeschreibung (ohne Anforderungen an Kandidaten und künftige Aufgaben, Datei Stellenbeschreibung.txt)
* Auflistung von Anforderungen der Firma (Datei Anforderungen.txt)
* Auflistung von künftigen Aufgaben (Datei Aufgaben.txt)

Alle Musterdateien mit oben angegebenen Namen müssen individuell angepasst bzw. durch eigene massgeschneiderte Varianten ersetzt werden (Regeln.md kann bleiben, wenn Sie diese Regeln nicht besser formulieren können).

Datei Lebenslauf-etc.md enthält alle für den Arbeitgeber wichtigen Informationen über Sie.
Sie besteht aus Kategorien ('# ' am Zeilenanfang), Unterkategorien ('## ' am Zeilenanfang, wird vom Programm als Titel für Hauptgedanken des Anschreibens verwendet) und Absätzen ('### ' am Zeilenanfang, wird vom Programm zum Erzeugen des Textes verwendet). Privatdate können Sie, damit sie den Cloud-Modellen nicht preisgegeben werden, durch fiktive ersetzen (Ernst Mustermann).

Datei Stellenbeschreibung.txt enthält Angaben über die Firma und Stelle, aber ohne die Listen der künftigen Aufgaben, die man separat in die Datei Aufgaben.txt, und Anforderungen an Kandidaten, die man separat in die Datei Anforderungen.txt ablegt. Alle Zeilen in Aufgaben.txt und Anforderungen.txt sollen durch leere Zeilen getrennt werden. Zum Erstellen der Dateien kann man C&P aus der Stellenanzeige benutzen.


Damit das Programm richtig fünktionieren kann, braucht man einen Docker Desktop, eine Ubuntu-WSL, wenn Sie unter Windows sind, einen Internet-Zugriff auf freie Cloud-LLMs und lokale LLMs.
Um den Zugriff auf freie Cloud-LLMs zu bekommen, brauchen Sie im Applikationsverzeichnis eine Datei .env anzulegen, die einen Inhalt vom Typ
```
GOOGLE_API_KEY=ABC...
MISTRAL_API_KEY=DEF...
GROQ_API_KEY=GHC...
```
hat, wo ABC..., DEF..., GHC... - ihre individuellen freien API-Keys sein müssen, die Sie nach der Anmeldung bei GOOGLE, MISTRAL und GROQ CLOUD CONSOLE erhalten.

In der Datei docker-compose.yml unter `# Optional: GPU Support aktivieren` sind darauffolgende Zeilen auskommentiert (mit '# ' vorne versehen). Das wurde gemacht, damit das Programm auch auf Servers ohne Grafikkarten funktionieren könnte (was aber deutlich länger dauert). Wenn Sie eine Grafikkarte >=4Gb besitzen, entfernen Sie dort diese '# '-Zeichen.

Starten Sie ihr Docker Desktop.
Wenn Sie unter Windows sind, starten Sie ihr Ubuntu WSL Terminal und stellen Sie sicher, dass Sie im Verzeichnis sind, wo sich generate_resume.py und andere obengenannten Dateien befinden.
Starten Sie die Dienste:
```
docker compose up --build
```
Ollama lädt die lokalen Modelle beim ersten Aufruf herunter, was beim Start der App zu einem Timeout führen kann. Es ist besser, das Modell vorher zu "pre-pullen" (selbst wenn die App mit Fehlern abgestürzt ist).
Öffnen Sie ein zweites Ubuntu WSL Terminal, während das erste `docker compose up` läuft, und führen Sie diesen Befehl aus, um die lokalem Modelle in das persistente Volume zu laden:
```
docker exec cover_letter-ollama-1 ollama pull mxbai-embed-large
docker exec cover_letter-ollama-1 ollama pull llama3.2
docker exec cover_letter-ollama-1 ollama pull gemma4:e4b
```
Sobald der Download abgeschlossen ist, können Sie den ersten Container-Run stoppen (STRG+C) und erneut `docker compose up --build` ausführen, um zu sehen, wie die Python-App erfolgreich mit dem Modell kommuniziert.

Wenn beim Aufruf `docker compose up --build` eine Fehlermeldung `failed to solve: error getting credentials - err: fork/exec /usr/bin/docker-credential-desktop.exe: exec format error, out: ''` entsteht, machen Sie folgendes:

```
mv ~/.docker/config.json ~/.docker/config.json.backup
rm ~/.docker/config.json
```

Die Erstellung eines Musteranschreibens passiert durch eine Reihe von Befragungen eines Agenten, der Sie vertritt und Ihr Lebenslauf als Grundlage für seine Antworte hat, durch einen Agenten, der die HR-Abteilung der in Frage kommenden Firma vertritt, und die Stellenbeschreibung, Aufgaben und Anforderungen als Basis hat. Dem Agenten, der Sie vertritt, steht ein "Vermittler von IT-Fachkräften" zur Seite, der ihm durch seine Kritik seiner Antworte hilft, diese unzuformulieren, damit sie mehr den Tatsachen im Lebenslauf und den Anforderungen des Punktes des Anschreibens entsprechen. Die Befragungen orientieren sich separat auf Anforderungen jedes in den Regeln festgelegten Punktes des Anschreibens. Bei der Fragestellung wird auf ehemalige Fragen und Antworte darauf orientiert. Dabei wird analysiert, ob die Befragung nicht einen falschen Weg genommen hat: Dann wird sie gestoppt und ab der letzten gelungenen Frage wiederaufgenommen. Am Ende wird für maximum drei so ausgeüllten Fragebögen zusammengefasst und die beste Zusammenfassung wird in die dem Punkt des Anschreibens entsprechende Datei im Unterverzeichnis app_output abgelegt.
Dort wird am Ende die Datei cover_letter_body.txt erzeugt, die eine Zusammenfassung von Dateien EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt, SCHLUSSSATZ.txt (früher erstellte Zusammenfassungen zu jedem Punkt des Anschreibens) darstellt und das Musteranschreiben ist. Wenn der Inhalt dieser Datei Ihnen nicht gefällt, können Sie die Applikation erneut starten, sodass auf Basis von Dateien EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt und SCHLUSSSATZ.txt eine neue Variante erstellt wird. Wenn Ihnen den Inhalt einer oder mehrerer Dateien unter EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt, SCHLUSSSATZ.txt nicht gefällt, können Sie diesen Inhalt manuell korrigieren oder die Datei komplett löschen, und den Vorgang wiederholen. 

Für eine komplett neue Bewerbung muss das Unterverzeichnis app_output leer sein.

Im Programm wurden einige Beispiele von https://github.com/benman1/generative_ai_with_langchain/blob/second_edition benutzt.