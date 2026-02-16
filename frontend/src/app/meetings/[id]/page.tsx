'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { getMeeting, updateMeetingStatus, FirestoreMeeting } from '@/services/firestoreService';
import { Button } from '@/components/ui/button';
import { Mic, Calendar, Users, ArrowLeft, Clock } from 'lucide-react';
import { format } from 'date-fns';

export default function MeetingPage() {
  const params = useParams();
  const meetingId = params.id as string;
  const router = useRouter();
  const { user } = useAuth();

  const [meeting, setMeeting] = useState<FirestoreMeeting | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!meetingId) return;
    getMeeting(meetingId)
      .then(setMeeting)
      .catch(() => setMeeting(null))
      .finally(() => setLoading(false));
  }, [meetingId]);

  const handleStartRecording = async () => {
    if (!meeting) return;

    // Update status to recording in Firestore
    await updateMeetingStatus(meetingId, 'recording');

    // Navigate to home page with auto-start recording
    sessionStorage.setItem('autoStartRecording', 'true');
    sessionStorage.setItem('activeMeetingId', meetingId);
    sessionStorage.setItem('activeMeetingTitle', meeting.title);
    router.push('/');
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-muted-foreground">Chargement...</p>
      </div>
    );
  }

  if (!meeting) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center space-y-3">
          <p className="text-muted-foreground">Réunion introuvable</p>
          <Button variant="outline" onClick={() => router.push('/')}>Retour</Button>
        </div>
      </div>
    );
  }

  const isOrganizer = user?.uid === meeting.organizerId;

  return (
    <div className="mx-auto max-w-lg space-y-6 p-6">
      <div className="flex items-center gap-3">
        <button onClick={() => router.back()} className="text-gray-500 hover:text-gray-700">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-2xl font-bold">{meeting.title}</h1>
      </div>

      <div className="space-y-4">
        {/* Status badge */}
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium ${
            meeting.status === 'planned'
              ? 'bg-blue-100 text-blue-700'
              : meeting.status === 'recording'
                ? 'bg-red-100 text-red-700'
                : 'bg-green-100 text-green-700'
          }`}>
            {meeting.status === 'planned' && <Clock className="w-3 h-3" />}
            {meeting.status === 'planned' ? 'Planifiée' : meeting.status === 'recording' ? 'En cours' : 'Terminée'}
          </span>
        </div>

        {/* Date */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Calendar className="w-4 h-4" />
          <span>Créée le {format(meeting.createdAt, 'dd/MM/yyyy à HH:mm')}</span>
        </div>

        {/* Participants */}
        {meeting.participants.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Users className="w-4 h-4" />
              <span>Participants ({meeting.participants.length})</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {meeting.participants.map((email) => (
                <span key={email} className="rounded-full bg-gray-100 px-3 py-1 text-xs">
                  {email}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        {meeting.status === 'planned' && isOrganizer && (
          <Button onClick={handleStartRecording} className="w-full" variant="green">
            <Mic className="w-4 h-4 mr-2" />
            Démarrer l'enregistrement
          </Button>
        )}

        {meeting.status === 'completed' && (
          <Button
            onClick={() => router.push(`/meeting-details?id=${meetingId}`)}
            className="w-full"
            variant="outline"
          >
            Voir le compte-rendu
          </Button>
        )}
      </div>
    </div>
  );
}
