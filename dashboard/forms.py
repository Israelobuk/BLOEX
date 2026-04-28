from django import forms


class ExplainForm(forms.Form):
    question = forms.CharField(
        max_length=500,
        widget=forms.TextInput(
            attrs={
                "placeholder": "Ask one sharp question about the source material...",
                "class": "w-full bg-slate-950/40 border border-slate-700 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none",
            }
        ),
    )
    context = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 14,
                "placeholder": "Paste the source material, notes, transcripts, or evidence here...",
                "class": "w-full bg-slate-950/40 border border-slate-700 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none",
            }
        )
    )
    base_url = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "http://localhost:11434",
                "class": "w-full bg-slate-950/40 border border-slate-700 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none",
            }
        ),
    )
    model = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "phi3:mini",
                "class": "w-full bg-slate-950/40 border border-slate-700 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none",
            }
        ),
    )
