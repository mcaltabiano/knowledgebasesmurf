# Knowledge base assistant smurf

![Screenshot 2025-01-12 182705](https://github.com/user-attachments/assets/47c8164a-ebac-4030-b360-dce8d454657a)

Silly project of a knowledge base assistant who answers questions about my docs.
Reads from pdf files under the "knowledge-base" folder.

Can set a custom prompt in `config.yaml`:
```yaml
assistant:
  custom_prompt: >
    Conversazione precedente:
    {chat_history}
    Contesto del sistema: Sei un assistente specializzato in ...
    Mantieni un tono professionale e fornisci risposte precise. Se non conosci la risposta rispondi semplicemente che non ti è possibile aiutare l'utente.
    Input attuale: {message}
```
