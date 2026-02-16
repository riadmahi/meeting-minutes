import {
  collection,
  doc,
  getDoc,
  getDocs,
  setDoc,
  updateDoc,
  deleteDoc,
  query,
  where,
  orderBy,
  onSnapshot,
  serverTimestamp,
  Timestamp,
  addDoc,
  writeBatch,
} from 'firebase/firestore';
import { db } from '@/lib/firebase';

// ─── Meeting types ───

export interface FirestoreMeeting {
  id: string;
  title: string;
  organizerId: string;
  participants: string[];
  status: 'planned' | 'recording' | 'completed';
  toolEnabled: boolean;
  folderPath?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface FirestoreTranscript {
  id: string;
  text: string;
  speaker?: string;
  timestamp: string;
  audioStartTime?: number;
  audioEndTime?: number;
  duration?: number;
}

export interface FirestoreSummary {
  id: string;
  status: 'pending' | 'completed' | 'failed';
  result?: string;
  createdAt: Date;
}

// ─── Meetings CRUD ───

const meetingsRef = collection(db, 'meetings');

export async function createMeeting(
  organizerId: string,
  title: string,
  participants: string[] = [],
): Promise<string> {
  const docRef = await addDoc(meetingsRef, {
    title,
    organizerId,
    participants,
    status: 'planned',
    toolEnabled: true,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  });
  return docRef.id;
}

export async function getMeeting(meetingId: string): Promise<FirestoreMeeting | null> {
  const snap = await getDoc(doc(db, 'meetings', meetingId));
  if (!snap.exists()) return null;
  return toMeeting(snap.id, snap.data());
}

export async function getUserMeetings(userId: string): Promise<FirestoreMeeting[]> {
  const q = query(
    meetingsRef,
    where('organizerId', '==', userId),
    orderBy('createdAt', 'desc'),
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => toMeeting(d.id, d.data()));
}

export function onUserMeetingsChanged(
  userId: string,
  callback: (meetings: FirestoreMeeting[]) => void,
) {
  const q = query(
    meetingsRef,
    where('organizerId', '==', userId),
    orderBy('createdAt', 'desc'),
  );
  return onSnapshot(q, (snap) => {
    callback(snap.docs.map((d) => toMeeting(d.id, d.data())));
  });
}

export async function updateMeetingTitle(meetingId: string, title: string): Promise<void> {
  await updateDoc(doc(db, 'meetings', meetingId), {
    title,
    updatedAt: serverTimestamp(),
  });
}

export async function updateMeetingStatus(
  meetingId: string,
  status: FirestoreMeeting['status'],
): Promise<void> {
  await updateDoc(doc(db, 'meetings', meetingId), {
    status,
    updatedAt: serverTimestamp(),
  });
}

export async function deleteMeeting(meetingId: string): Promise<void> {
  await deleteDoc(doc(db, 'meetings', meetingId));
}

// ─── Transcripts (subcollection) ───

export async function addTranscript(
  meetingId: string,
  transcript: Omit<FirestoreTranscript, 'id'>,
): Promise<string> {
  const ref = collection(db, 'meetings', meetingId, 'transcripts');
  const docRef = await addDoc(ref, transcript);
  return docRef.id;
}

export async function getTranscripts(meetingId: string): Promise<FirestoreTranscript[]> {
  const ref = collection(db, 'meetings', meetingId, 'transcripts');
  const q = query(ref, orderBy('audioStartTime', 'asc'));
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() } as FirestoreTranscript));
}

export async function addTranscriptsBatch(
  meetingId: string,
  transcripts: Omit<FirestoreTranscript, 'id'>[],
): Promise<void> {
  const ref = collection(db, 'meetings', meetingId, 'transcripts');
  await Promise.all(transcripts.map((t) => addDoc(ref, t)));
}

// ─── Summaries (subcollection) ───

export async function saveSummary(
  meetingId: string,
  result: string,
): Promise<void> {
  await setDoc(doc(db, 'meetings', meetingId, 'summaries', 'latest'), {
    status: 'completed',
    result,
    createdAt: serverTimestamp(),
  });
}

export async function getSummary(meetingId: string): Promise<FirestoreSummary | null> {
  const snap = await getDoc(doc(db, 'meetings', meetingId, 'summaries', 'latest'));
  if (!snap.exists()) return null;
  const data = snap.data();
  return {
    id: snap.id,
    status: data.status,
    result: data.result,
    createdAt: toDate(data.createdAt),
  };
}

// ─── Tasks (subcollection of meetings) ───

export interface FirestoreTask {
  id: string;
  meetingId: string;
  description: string;
  assignee?: string;
  dueDate?: string;
  status: 'pending' | 'in_progress' | 'completed';
  priority?: 'low' | 'medium' | 'high';
  createdAt: Date;
  updatedAt: Date;
}

export async function addTask(
  meetingId: string,
  task: Omit<FirestoreTask, 'id' | 'meetingId' | 'createdAt' | 'updatedAt'>,
): Promise<string> {
  const ref = collection(db, 'meetings', meetingId, 'tasks');
  const docRef = await addDoc(ref, {
    ...task,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  });
  return docRef.id;
}

export async function addTasksBatch(
  meetingId: string,
  tasks: Omit<FirestoreTask, 'id' | 'meetingId' | 'createdAt' | 'updatedAt'>[],
): Promise<void> {
  const batch = writeBatch(db);
  const ref = collection(db, 'meetings', meetingId, 'tasks');
  for (const task of tasks) {
    const docRef = doc(ref);
    batch.set(docRef, {
      ...task,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
  }
  await batch.commit();
}

export async function getMeetingTasks(meetingId: string): Promise<FirestoreTask[]> {
  const ref = collection(db, 'meetings', meetingId, 'tasks');
  const q = query(ref, orderBy('createdAt', 'asc'));
  const snap = await getDocs(q);
  return snap.docs.map((d) => toTask(d.id, meetingId, d.data()));
}

export async function updateTaskStatus(
  meetingId: string,
  taskId: string,
  status: FirestoreTask['status'],
): Promise<void> {
  await updateDoc(doc(db, 'meetings', meetingId, 'tasks', taskId), {
    status,
    updatedAt: serverTimestamp(),
  });
}

export async function deleteTask(meetingId: string, taskId: string): Promise<void> {
  await deleteDoc(doc(db, 'meetings', meetingId, 'tasks', taskId));
}

export function onMeetingTasksChanged(
  meetingId: string,
  callback: (tasks: FirestoreTask[]) => void,
) {
  const ref = collection(db, 'meetings', meetingId, 'tasks');
  const q = query(ref, orderBy('createdAt', 'asc'));
  return onSnapshot(q, (snap) => {
    callback(snap.docs.map((d) => toTask(d.id, meetingId, d.data())));
  });
}

/** Get all tasks across all meetings for a given user (by assignee email) */
export async function getUserTasks(userEmail: string): Promise<FirestoreTask[]> {
  // First get all meetings the user organizes or participates in
  const orgQuery = query(meetingsRef, where('organizerId', '==', userEmail));
  const partQuery = query(meetingsRef, where('participants', 'array-contains', userEmail));

  const [orgSnap, partSnap] = await Promise.all([getDocs(orgQuery), getDocs(partQuery)]);

  const meetingIds = new Set<string>();
  orgSnap.docs.forEach((d) => meetingIds.add(d.id));
  partSnap.docs.forEach((d) => meetingIds.add(d.id));

  // Fetch tasks from each meeting
  const allTasks: FirestoreTask[] = [];
  await Promise.all(
    Array.from(meetingIds).map(async (mId) => {
      const tasks = await getMeetingTasks(mId);
      allTasks.push(...tasks);
    }),
  );

  return allTasks.sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime());
}

// ─── Helpers ───

function toDate(ts: Timestamp | null | undefined): Date {
  return ts?.toDate() ?? new Date();
}

function toMeeting(id: string, data: Record<string, unknown>): FirestoreMeeting {
  return {
    id,
    title: data.title as string,
    organizerId: data.organizerId as string,
    participants: (data.participants as string[]) || [],
    status: (data.status as FirestoreMeeting['status']) || 'planned',
    toolEnabled: (data.toolEnabled as boolean) ?? true,
    folderPath: data.folderPath as string | undefined,
    createdAt: toDate(data.createdAt as Timestamp),
    updatedAt: toDate(data.updatedAt as Timestamp),
  };
}

function toTask(id: string, meetingId: string, data: Record<string, unknown>): FirestoreTask {
  return {
    id,
    meetingId,
    description: (data.description as string) || '',
    assignee: data.assignee as string | undefined,
    dueDate: data.dueDate as string | undefined,
    status: (data.status as FirestoreTask['status']) || 'pending',
    priority: data.priority as FirestoreTask['priority'],
    createdAt: toDate(data.createdAt as Timestamp),
    updatedAt: toDate(data.updatedAt as Timestamp),
  };
}
