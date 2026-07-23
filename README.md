Das Programm erstellt ein Beispiel eines Initiativbriefes/Anschreibens auf Basis von
* Ihrem Lebenslauf, einschließlich Zeugnisse, Zertifikate und Diplome (Datei Lebenslauf-etc.md)
* Regeln für Initiativbriefe/Anschreiben (Datei Regeln.md) 
* Firmen/Stellenbeschreibung (ohne Anforderungen an Kandidaten und künftige Aufgaben, Datei Stellenbeschreibung.txt)
* Auflistung von Anforderungen der Firma (Datei Anforderungen.txt)
* Auflistung von künftigen Aufgaben (Datei Aufgaben.txt)

Alle Musterdateien mit oben angegebenen Namen müssen individuell angepasst bzw. durch eigene massgeschneiderte Varianten ersetzt werden (Regeln.md kann bleiben, wenn Sie diese Regeln nicht besser formulieren können).

Datei Lebenslauf-etc.md enthält alle für den Arbeitgeber wichtigen Informationen über Sie.
Sie besteht aus Kategorien ('# ' am Zeilenanfang, der Titel wird im Programm nicht eingelesen), Unterkategorien ('## ' am Zeilenanfang, dieser Titel wird vom Programm als Titel für Hauptgedanken des Anschreibens verwendet) und Absätzen ('### ' am Zeilenanfang, der Titel wird im Programm nicht eingelesen, der Inhalt darunter wird vom Programm zum Erzeugen des Textes verwendet). Privatdaten können Sie, damit sie den Cloud-Modellen nicht preisgegeben werden, durch fiktive ersetzen (z.B. Ernst Mustermann).

Datei Stellenbeschreibung.txt enthält Angaben über die Firma und Stelle, aber ohne die Listen der künftigen Aufgaben, die man separat in die Datei Aufgaben.txt, und Anforderungen an Kandidaten, die man separat in die Datei Anforderungen.txt ablegt. Alle Zeilen in Aufgaben.txt und Anforderungen.txt sollen durch leere Zeilen getrennt werden. Zum Erstellen der Dateien kann man Copy&Paste aus der Stellenanzeige benutzen.


Damit das Programm richtig funktionieren kann, braucht man einen Docker Desktop, eine Ubuntu-WSL, wenn Sie unter Windows sind, einen Internet-Zugriff auf freie Cloud-LLMs und lokale LLMs.
Um den Zugriff auf freie Cloud-LLMs zu bekommen, brauchen Sie im Applikationsverzeichnis eine Datei .env anzulegen, die einen Inhalt vom Typ
```
GOOGLE_API_KEY=ABC...
MISTRAL_API_KEY=DEF...
GROQ_API_KEY=GHC...
```
hat, wo ABC..., DEF..., GHC... - ihre individuellen freien API-Keys sein müssen, die Sie nach der Anmeldung bei GOOGLE, MISTRAL und GROQ CLOUD CONSOLE erhalten.

Wenn Sie keine Grafikkarte >=4G besitzen, kommentieren sie in der Datei docker-compose.yml unter `# Optional: GPU Support aktivieren` Zeilen bis einschließlich `capabilities: [gpu]` aus (auskommentieren bedeutet mit # vorne versehen), damit das Programm auch auf Servers ohne Grafikkarten funktionieren könnte (was aber deutlich länger dauert).

Starten Sie ihr Docker Desktop.
Wenn Sie unter Windows sind, starten Sie ihr Ubuntu WSL Terminal und stellen Sie sicher, dass Sie im Verzeichnis sind, wo sich generate_resume.py mit dem Programm und andere oben genannte Dateien befinden.
Starten Sie die Dienste:
```
docker compose up --build
```
Ollama lädt die lokalen Modelle beim ersten Aufruf herunter, was beim Start der App zu einem Timeout führen kann. Es kann auch sein, dass die Modelle nicht runtergeladen sind, dann passiert ein Fehler. Das Modell ist vorher zu "pre-pullen" (selbst wenn die App mit Fehlern abgestürzt ist).
Öffnen Sie ein zweites Ubuntu (WSL) Terminal, während das erste `docker compose up` läuft oder nach der Fehlermeldung wartet, und führen Sie diese Befehle aus, um die lokalen Modelle in das persistente Volume zu laden:
```
docker exec cover_letter-ollama-1 ollama pull mxbai-embed-large
docker exec cover_letter-ollama-1 ollama pull llama3.2:3b
docker exec cover_letter-ollama-1 ollama pull gemma4:e4b
docker exec cover_letter-ollama-1 ollama pull mistral
```
Sobald der Download abgeschlossen ist, können Sie den ersten Container-Run stoppen (STRG+C) und erneut `docker compose up --build` ausführen, um zu sehen, wie die Python-App erfolgreich mit dem Modell kommuniziert.

Wenn beim Aufruf `docker compose up --build` eine Fehlermeldung `failed to solve: error getting credentials - err: fork/exec /usr/bin/docker-credential-desktop.exe: exec format error, out: ''` entsteht, machen Sie auf dem zweiten Ubuntu (WSL) Terminal folgendes:

```
mv ~/.docker/config.json ~/.docker/config.json.backup
rm ~/.docker/config.json
```
Dann den ersten Container-Run auf dem ersten Ubuntu (WSL) Terminal stoppen (STRG+C) und erneut `docker compose up --build` ausführen.

Die Erstellung eines Musteranschreibens passiert durch eine Reihe von Befragungen eines Agenten, der Sie vertritt, und Ihr Lebenslauf als Grundlage für seine Antworte hat, durch einen anderen Agenten, der die HR-Abteilung der in Frage kommenden Firma vertritt, und die Stellenbeschreibung, Aufgaben und Anforderungen als Basis hat. 
Dem Agenten, der Sie vertritt, steht ein LangChain in der Rolle "Vermittler von IT-Fachkräften" zur Seite, der ihm durch seine Kritik der gegebenen Antwort hilft, diese umzuformulieren, damit sie mehr den Tatsachen im Lebenslauf und den Anforderungen des Punktes des Anschreibens entsprächen. 
Die Befragungen orientieren sich separat auf Anforderungen jedes in den Regeln festgelegten Punktes des Anschreibens. 
Bei der Fragestellung wird auf ehemalige Fragen und Antworte darauf orientiert. 
Dabei wird bei jeder beantworteten Frage analysiert, ob eine bessere Frage möglich ist: Dann wird sie gestellt und ein neuer Fragebogen entsteht. 
Am Ende wird für maximal drei (Konfiguration) so ausgefüllter Fragebögen eine Zusammenfassung erstellt und die beste davon wird in die dem Punkt des Anschreibens entsprechende Datei im Unterverzeichnis app_output geschrieben.
Dort wird am Ende auch die Datei cover_letter_body.txt erzeugt, die eine Zusammenfassung von Dateien EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt, SCHLUSSSATZ.txt (früher erstellte Zusammenfassungen zu jedem Punkt des Anschreibens) darstellt, und das endgültige Musteranschreiben enthält. 
Wenn der Inhalt dieser Datei Ihnen nicht gefällt, können Sie die Applikation erneut starten (nach STRG+C im ersten Terminal), sodass auf Basis von Dateien EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt und SCHLUSSSATZ.txt eine neue Variante Ihres Initiativbriefes erstellt wird. 
Wenn Ihnen der Inhalt einer oder mehrerer Dateien unter EINLEITUNGSSATZ.txt, HAUPTTEIL-Eigenmarketing.txt, HAUPTTEIL-Unternehmensbezug.txt, SCHLUSSSATZ.txt nicht gefällt, können Sie diesen manuell korrigieren oder die Datei komplett löschen, und den Vorgang mit `docker compose up --build` wiederholen. 

Für eine komplett neue Bewerbung muss das Unterverzeichnis app_output leer sein!

Im Programm wurden einige Beispiele von https://github.com/benman1/generative_ai_with_langchain/blob/second_edition benutzt.