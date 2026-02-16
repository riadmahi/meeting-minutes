'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { createMeeting } from '@/services/firestoreService';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { toast } from 'sonner';
import { ArrowLeft, Plus, X } from 'lucide-react';

export default function NewMeetingPage() {
  const router = useRouter();
  const { user } = useAuth();

  const [title, setTitle] = useState('');
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [participantEmail, setParticipantEmail] = useState('');
  const [participants, setParticipants] = useState<string[]>([]);
  const [toolEnabled, setToolEnabled] = useState(true);
  const [agenda, setAgenda] = useState('');
  const [creating, setCreating] = useState(false);

  const addParticipant = () => {
    const email = participantEmail.trim().toLowerCase();
    if (!email || !email.includes('@')) return;
    if (participants.includes(email)) return;
    setParticipants([...participants, email]);
    setParticipantEmail('');
  };

  const removeParticipant = (email: string) => {
    setParticipants(participants.filter((p) => p !== email));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addParticipant();
    }
  };

  const handleCreate = async () => {
    if (!user) return;
    if (!title.trim()) {
      toast.error('Le titre est requis');
      return;
    }

    setCreating(true);
    try {
      const meetingId = await createMeeting(user.uid, title.trim(), participants);
      toast.success('Réunion créée');
      router.push(`/meeting-details?id=${meetingId}`);
    } catch (err) {
      console.error('Failed to create meeting:', err);
      toast.error('Erreur lors de la création');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="mx-auto max-w-lg space-y-6 p-6">
      <div className="flex items-center gap-3">
        <button onClick={() => router.back()} className="text-gray-500 hover:text-gray-700">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-2xl font-bold">Nouvelle réunion</h1>
      </div>

      <div className="space-y-4">
        {/* Title */}
        <div className="space-y-2">
          <Label htmlFor="title">Titre</Label>
          <Input
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Ex: Standup hebdomadaire"
          />
        </div>

        {/* Date & Time */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label htmlFor="date">Date</Label>
            <Input
              id="date"
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="time">Heure</Label>
            <Input
              id="time"
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
            />
          </div>
        </div>

        {/* Participants */}
        <div className="space-y-2">
          <Label>Participants</Label>
          <div className="flex gap-2">
            <Input
              value={participantEmail}
              onChange={(e) => setParticipantEmail(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="email@exemple.com"
              type="email"
            />
            <Button type="button" variant="outline" size="icon" onClick={addParticipant}>
              <Plus className="w-4 h-4" />
            </Button>
          </div>
          {participants.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-2">
              {participants.map((email) => (
                <span
                  key={email}
                  className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-3 py-1 text-sm"
                >
                  {email}
                  <button onClick={() => removeParticipant(email)} className="text-gray-400 hover:text-red-500">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Agenda */}
        <div className="space-y-2">
          <Label htmlFor="agenda">Ordre du jour</Label>
          <Textarea
            id="agenda"
            value={agenda}
            onChange={(e) => setAgenda(e.target.value)}
            placeholder="Points à aborder..."
            rows={3}
          />
        </div>

        {/* Tool toggle */}
        <div className="flex items-center justify-between rounded-lg border p-3">
          <div>
            <p className="text-sm font-medium">Activer l'outil de transcription</p>
            <p className="text-xs text-muted-foreground">Enregistrer et transcrire automatiquement</p>
          </div>
          <Switch checked={toolEnabled} onCheckedChange={setToolEnabled} />
        </div>

        {/* Submit */}
        <Button onClick={handleCreate} disabled={creating} className="w-full">
          {creating ? 'Création...' : 'Créer la réunion'}
        </Button>
      </div>
    </div>
  );
}
