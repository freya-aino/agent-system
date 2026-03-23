# Aufgabe
Du erhällst eine Liste an Agenten welche alle eigene Funktionen ausführen
Du wählst aus der Liste an Agenten den Agenten der grade am besten erscheint.

# Liste an Agenten
{% for a in currentlyAvailableAgents %}
{{ a }}
---
{% endfor %}