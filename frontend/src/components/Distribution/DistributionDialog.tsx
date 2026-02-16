'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog';
import { VisuallyHidden } from '@/components/ui/visually-hidden';
import { FileDown, Link2, Mail, Plus, Send, X } from 'lucide-react';
import { toast } from 'sonner';
import { exportSummaryToPdf } from '@/services/pdfExportService';
import { getMeeting } from '@/services/firestoreService';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface DistributionDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  meetingId: string;
  meetingTitle: string;
  createdAt?: string;
  markdown: string;
  participants?: string[];
}

export function DistributionDialog({
  open,
  onOpenChange,
  meetingId,
  meetingTitle,
  createdAt,
  markdown,
  participants = [],
}: DistributionDialogProps) {
  const [recipientEmail, setRecipientEmail] = useState('');
  const [recipients, setRecipients] = useState<string[]>(participants);
  const [isSending, setIsSending] = useState(false);

  const addRecipient = () => {
    const email = recipientEmail.trim().toLowerCase();
    if (!email || !email.includes('@')) return;
    if (recipients.includes(email)) return;
    setRecipients([...recipients, email]);
    setRecipientEmail('');
  };

  const removeRecipient = (email: string) => {
    setRecipients(recipients.filter((r) => r !== email));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addRecipient();
    }
  };

  const handleExportPdf = async () => {
    try {
      const meeting = await getMeeting(meetingId).catch(() => null);
      await exportSummaryToPdf({
        title: meetingTitle,
        markdown,
        meetingId,
        createdAt,
        participants: meeting?.participants,
      });
      toast.success('PDF exporté');
    } catch {
      toast.error("Erreur lors de l'export PDF");
    }
  };

  const handleCopyLink = async () => {
    const link = `https://app.remedee.com/cr/${meetingId}`;
    try {
      await navigator.clipboard.writeText(link);
      toast.success('Lien copié');
    } catch {
      toast.error('Impossible de copier le lien');
    }
  };

  const handleSendEmail = async () => {
    if (recipients.length === 0) {
      toast.error('Ajoutez au moins un destinataire');
      return;
    }

    setIsSending(true);
    try {
      const subject = encodeURIComponent(`Compte-rendu : ${meetingTitle}`);
      const truncated = markdown.substring(0, 2000);
      const body = encodeURIComponent(
        `Bonjour,\n\nVoici le compte-rendu de la réunion "${meetingTitle}".\n\n${truncated}${markdown.length > 2000 ? '\n\n[Contenu tronqué]' : ''}\n\n---\nGénéré par Remedee`,
      );
      const to = recipients.join(',');
      window.open(`mailto:${to}?subject=${subject}&body=${body}`, '_blank');
      toast.success('Client email ouvert');
      onOpenChange(false);
    } catch {
      toast.error("Erreur lors de l'envoi");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
        <VisuallyHidden>
          <DialogTitle>Distribuer le compte-rendu</DialogTitle>
        </VisuallyHidden>

        <div className="flex items-center justify-between pb-3 border-b">
          <h2 className="text-lg font-semibold">Distribuer le compte-rendu</h2>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleExportPdf}>
              <FileDown className="w-4 h-4 mr-1" />
              PDF
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopyLink}>
              <Link2 className="w-4 h-4 mr-1" />
              Lien
            </Button>
          </div>
        </div>

        {/* Recipients */}
        <div className="space-y-2 py-3 border-b">
          <Label>Destinataires</Label>
          <div className="flex gap-2">
            <Input
              value={recipientEmail}
              onChange={(e) => setRecipientEmail(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="email@exemple.com"
              type="email"
            />
            <Button variant="outline" size="icon" onClick={addRecipient}>
              <Plus className="w-4 h-4" />
            </Button>
          </div>
          {recipients.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-2">
              {recipients.map((email) => (
                <span
                  key={email}
                  className="inline-flex items-center gap-1 rounded-full bg-blue-50 border border-blue-200 px-3 py-1 text-xs text-blue-700"
                >
                  {email}
                  <button onClick={() => removeRecipient(email)} className="text-blue-400 hover:text-red-500">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Preview */}
        <div className="flex-1 overflow-y-auto min-h-0 py-3">
          <Label className="mb-2 block">Aperçu du compte-rendu</Label>
          <div className="prose prose-sm max-w-none border rounded-lg p-4 bg-gray-50 max-h-[300px] overflow-y-auto">
            <h3 className="text-base font-semibold mb-2">{meetingTitle}</h3>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {markdown}
            </ReactMarkdown>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-3 border-t">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Fermer
          </Button>
          <Button onClick={handleSendEmail} disabled={isSending || recipients.length === 0}>
            <Send className="w-4 h-4 mr-2" />
            {isSending ? 'Envoi...' : `Envoyer (${recipients.length})`}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
