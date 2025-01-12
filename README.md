# Knowledge base assistant Smurf

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

Comes from an exercise from a [Ed Donner's Udemy course](https://www.udemy.com/share/10bOXH3@6jbJpbt8suPadW9u7KDkk2UNCJp1OCCOMhPImzx5UdaOk3rIvarBtjoa5M32EnW_dg==/)

It uses [ollama](https://ollama.com/) to run local llms.
