'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { updateUserProfile } from '@/services/firebaseService';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';

export default function ProfilePage() {
  const { user, profile } = useAuth();
  const [displayName, setDisplayName] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (profile) {
      setDisplayName(profile.displayName || '');
    }
  }, [profile]);

  if (!user || !profile) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-muted-foreground">Chargement du profil...</p>
      </div>
    );
  }

  const handleSave = async () => {
    if (!displayName.trim()) return;
    setSaving(true);
    try {
      await updateUserProfile(user.uid, { displayName: displayName.trim() });
      toast.success('Profil mis à jour');
    } catch {
      toast.error('Erreur lors de la mise à jour');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-lg space-y-6 p-6">
      <h1 className="text-2xl font-bold">Mon profil</h1>

      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Email</label>
          <Input value={user.email || ''} disabled />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Nom</label>
          <Input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Votre nom"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Rôle</label>
          <Input value={profile.role === 'organizer' ? 'Organisateur' : 'Participant'} disabled />
        </div>

        <Button onClick={handleSave} disabled={saving}>
          {saving ? 'Enregistrement...' : 'Enregistrer'}
        </Button>
      </div>
    </div>
  );
}
