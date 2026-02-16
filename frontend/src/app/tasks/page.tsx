'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useSidebar } from '@/components/Sidebar/SidebarProvider';
import {
  FirestoreTask,
  getUserTasks,
  updateTaskStatus,
} from '@/services/firestoreService';
import { Button } from '@/components/ui/button';
import { ArrowLeft, CheckCircle2, Circle, Clock, ListTodo, Loader2 } from 'lucide-react';

type FilterStatus = 'all' | 'pending' | 'in_progress' | 'completed';

export default function TasksPage() {
  const router = useRouter();
  const { user } = useAuth();
  const { meetings } = useSidebar();

  const [tasks, setTasks] = useState<FirestoreTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterStatus>('all');

  const fetchTasks = useCallback(async () => {
    if (!user?.email) return;
    setLoading(true);
    try {
      const allTasks = await getUserTasks(user.email);
      setTasks(allTasks);
    } catch (err) {
      console.error('Failed to fetch tasks:', err);
    } finally {
      setLoading(false);
    }
  }, [user?.email]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const handleToggleStatus = async (task: FirestoreTask) => {
    const newStatus = task.status === 'completed' ? 'pending' : 'completed';
    try {
      await updateTaskStatus(task.meetingId, task.id, newStatus);
      setTasks((prev) =>
        prev.map((t) =>
          t.id === task.id ? { ...t, status: newStatus, updatedAt: new Date() } : t,
        ),
      );
    } catch (err) {
      console.error('Failed to update task:', err);
    }
  };

  const getMeetingTitle = (meetingId: string) => {
    const meeting = meetings.find((m) => m.id === meetingId);
    return meeting?.title || 'Réunion inconnue';
  };

  const filteredTasks = tasks.filter((t) => {
    if (filter === 'all') return true;
    return t.status === filter;
  });

  const pendingCount = tasks.filter((t) => t.status === 'pending').length;
  const completedCount = tasks.filter((t) => t.status === 'completed').length;

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div className="flex items-center gap-3">
        <button onClick={() => router.back()} className="text-gray-500 hover:text-gray-700">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <ListTodo className="w-6 h-6 text-blue-500" />
        <h1 className="text-2xl font-bold">Mes tâches</h1>
      </div>

      {/* Stats */}
      <div className="flex gap-4 text-sm">
        <span className="text-muted-foreground">
          {pendingCount} en attente
        </span>
        <span className="text-muted-foreground">
          {completedCount} terminée{completedCount > 1 ? 's' : ''}
        </span>
      </div>

      {/* Filters */}
      <div className="flex gap-2">
        {([
          { value: 'all', label: 'Toutes' },
          { value: 'pending', label: 'En attente' },
          { value: 'in_progress', label: 'En cours' },
          { value: 'completed', label: 'Terminées' },
        ] as { value: FilterStatus; label: string }[]).map((f) => (
          <Button
            key={f.value}
            variant={filter === f.value ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter(f.value)}
          >
            {f.label}
          </Button>
        ))}
      </div>

      {/* Task list */}
      {filteredTasks.length === 0 ? (
        <div className="text-center py-12">
          <ListTodo className="w-12 h-12 mx-auto text-gray-300 mb-3" />
          <p className="text-muted-foreground">
            {filter === 'all' ? 'Aucune tâche pour le moment' : 'Aucune tâche avec ce filtre'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredTasks.map((task) => (
            <div
              key={task.id}
              className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                task.status === 'completed'
                  ? 'bg-gray-50 border-gray-200'
                  : 'bg-white border-gray-200 hover:border-gray-300'
              }`}
            >
              <button
                onClick={() => handleToggleStatus(task)}
                className="mt-0.5 flex-shrink-0"
              >
                {task.status === 'completed' ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                ) : task.status === 'in_progress' ? (
                  <Clock className="w-5 h-5 text-orange-500" />
                ) : (
                  <Circle className="w-5 h-5 text-gray-300 hover:text-gray-500" />
                )}
              </button>

              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm ${
                    task.status === 'completed'
                      ? 'line-through text-gray-400'
                      : 'text-gray-900'
                  }`}
                >
                  {task.description}
                </p>

                <div className="flex flex-wrap gap-2 mt-1">
                  {/* Meeting link */}
                  <button
                    onClick={() => router.push(`/meeting-details?id=${task.meetingId}`)}
                    className="text-xs text-blue-500 hover:underline"
                  >
                    {getMeetingTitle(task.meetingId)}
                  </button>

                  {/* Assignee */}
                  {task.assignee && (
                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                      {task.assignee}
                    </span>
                  )}

                  {/* Due date */}
                  {task.dueDate && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      new Date(task.dueDate) < new Date() && task.status !== 'completed'
                        ? 'bg-red-100 text-red-600'
                        : 'bg-gray-100 text-gray-500'
                    }`}>
                      {new Date(task.dueDate).toLocaleDateString('fr-FR')}
                    </span>
                  )}

                  {/* Priority */}
                  {task.priority && task.priority !== 'medium' && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      task.priority === 'high'
                        ? 'bg-red-100 text-red-600'
                        : 'bg-blue-100 text-blue-600'
                    }`}>
                      {task.priority === 'high' ? 'Haute' : 'Basse'}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
