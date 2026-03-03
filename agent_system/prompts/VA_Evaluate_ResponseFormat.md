# Aufgabe
Bewerte den Gedankengang anhand der User Frage und der Dokument Elemente.
Bewerte die Antwort anhand des Gedankengangs.

# User Frage

{{ userQuestion }}

# Dokument Elemente

{% for d in documentElements %}
- {{ d }}
{% endfor %}