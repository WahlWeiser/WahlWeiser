import os
import streamlit as st
import pandas as pd
import openai
import re
from typing import Optional
import numpy as np
import asyncio
import json


openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.AsyncOpenAI()


parties = ["CDU", "SPD", "Grüne", "FDP", "Linke", "AfD"]
parties_long = {
    "CDU": "Christlich Demokratische Union",
    "SPD": "Sozialdemokratische Partei Deutschlands",
    "Grüne": "Bündnis 90/Die Grünen",
    "FDP": "Freie Demokratische Partei",
    "Linke": "Die Linke",
    "AfD": "Alternative für Deutschland",
}
SYSTEM_PROMPT = """
Du bewertest, wie gut eine Aussage zu den deutschen politischen Partein CDU, SPD, Grüne, FDP, Linke, und AfD passt. Gebe je eine Prozentzahl an, wie sehr die Parteien der gegebenen Aussage zustimmen. Gebe die Antwort in JSON. Gebe nichts weiteres aus.

Beispiel: {"CDU":50,"SPD":50,"Grüne":50,"FDP":50,"Linke":50,"AfD":50}
"""


async def evaluate(statement: str) -> Optional[int]:
    response = None
    try:
        response = await openai_client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': statement},
            ],
            temperature = 0.0,
            response_format = { "type": "json_object" }
        )
        response = response.choices[0].message.content
        print(response)
        response = json.loads(response)
        return response, None
    except Exception as e:
        print(e)
        return response, e

async def main():
    st.set_page_config(page_title='WahlWeiser', page_icon='logo.png')
    st.title("WahlWeiser")
    st.write("Welche Partei passt zu dir? Gebe eine Aussage ein und eine KI bewertet, wie gut diese Aussage zu den großen deutschen Parteien passt.")

    if "data" not in st.session_state:
        st.session_state["data"] = {}

    for statement, values in st.session_state.data.items():
        msg = st.chat_message('')
        msg.write(f'**{statement}**')
        msg.bar_chart(pd.DataFrame.from_dict(values, orient='index'))

    if statement := st.chat_input():
        statement = statement.strip()
        msg = st.chat_message("")
        msg.write(f'**{statement}**')
        with st.spinner('Die KI denkt nach...'):
            values, error = await evaluate(statement)
        if error is None:
            st.session_state.data[statement] = values
            print(values)
            msg.bar_chart(pd.DataFrame.from_dict(values, orient='index'))
        else:
            msg.error(f"Fehler bei der Bewertung ({type(error).__name__}). Bitte versuche es erneut.")
            with st.expander("Fehlermeldung anzeigen"):
                st.write(f'{error}')
                st.write(f'Response: `{values}`')


    with st.sidebar:
        st.image('logo.png', width=50)
        st.markdown('<style>img {margin-top:-5em};</style>', unsafe_allow_html=True)  # This is a hack and only works because we have only one image here
        
        st.write("Wie gut passen die Aussagen zu den Parteien?")
        values = st.session_state.data if len(st.session_state.data) > 0 else {'': {party: 50 for party in parties}}
        data = pd.DataFrame.from_dict(values, orient='index', columns=parties)
        st.bar_chart(data.mean())

        _, center_col, _ = st.columns([1, 1, 1])
        center_col.button("Reset", on_click=lambda: st.session_state.clear(), help="Löscht alle bisherigen Aussagen und Bewertungen.")
        st.empty()
        st.write("## Über WahlWeiser:")
        st.write('Dieses Projekt ist nur ein Experiment und erhebt keine Ansprüche auf Richtigkeit. Zur Bewertung der Aussagen wird die KI GPT-3.5-turbo von OpenAI verwendet. Die Bewertungen sind daher nicht unbedingt repräsentativ für die Parteien.')
        st.write('Diese Seite wird über die [Streamlit Community Cloud](https://streamlit.io/cloud) gehostet, die Eingegebenen Aussagen werden also auf Streamlit Servern verarbeitet und zur Bewertung an OpenAI übermittelt. Gebe daher keine privaten Daten ein.')
        st.write(f"Github: [Wahlweiser/WahlWeiser](https://github.com/Wahlweiser/WahlWeiser)")


if __name__ == "__main__":
    asyncio.run(main())
