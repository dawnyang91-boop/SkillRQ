# api_bank Code Cards

## Summary

| Metric | Value |
|---|---:|
| `assignment_count` | 101 |
| `unique_semantic_code_paths` | 101 |
| `unique_l1_codes` | 10 |
| `code_purity` | 1.000000 |
| `code_usage_entropy` | 1.000000 |
| `category_alignment` | 1.000000 |
| `role_alignment` | 1.000000 |
| `code_collapse_rate` | 0.009901 |

## Top L1 Codes

| Code | Label | Count | Examples |
|---|---|---:|---|
| `L1-label-927fa0e9` | 获取信息 - 数据库查询 | 30 | AIConferenceSearch, CheckTimeConflict, CompanyAnnualReport, CompanyBusinessScope |
| `L1-label-d50b723e` | 对外影响 - 数据库操作 - 增 | 12 | AddAgenda, AddAlarm, AddMeeting, AddReminder |
| `L1-label-823536a6` | 对外影响 - 数据库操作 - 删 | 12 | CancelHotelBooking, CancelRegistration, CancelTicket, CancelTimedSwitch |
| `L1-ai-f2c554b6` | 其他AI模型 | 11 | DocumentQA, IdentifySong, ImageCaption, ObjectDetection |
| `L1-label-e284b5a3` | 对外影响 - 数据库操作 - 查 | 11 | ForgotPassword, QueryAgenda, QueryAlarm, QueryBankAccount |
| `L1-label-b45672c8` | 对外影响 - 数据库操作 - 改 | 11 | ModifyAgenda, ModifyAlarm, ModifyBankAccount, ModifyHotelBooking |
| `L1-label-ab191893` | 对外影响 - 数据库操作 | 6 | AssignPriority, GetUserToken, Memo, RecordHealthData |
| `L1-label-f294435b` | 对外影响 - 实时通信 | 6 | ControlDevice, PlayMusic, ReceiveEmail, SendEmail |
| `L1-label-ad0b9ef9` | 获取信息 - 工具 | 1 | Calculator |
| `L1-label-66ca76a3` | 获取信息 - 外界信息查询 | 1 | GetToday |

## Top Semantic Paths

| Semantic ID | Count | Examples |
|---|---:|---|
| `L1-label-d50b723e/L2-add_agenda-d9b6af5e/L3-finalize-3244ae15/L4-method_unknown_multi_input_4_out-a6f9314c` | 1 | AddAgenda |
| `L1-label-d50b723e/L2-add_alarm-a97ea9c8/L3-finalize-3244ae15/L4-method_unknown_input_2_optional_-e01f8d3c` | 1 | AddAlarm |
| `L1-label-d50b723e/L2-add_meeting-8afcf06b/L3-finalize-3244ae15/L4-method_unknown_input_3_optional_-fefdb3de` | 1 | AddMeeting |
| `L1-label-d50b723e/L2-add_reminder-7accfca6/L3-finalize-3244ae15/L4-method_unknown_input_3_optional_-fefdb3de` | 1 | AddReminder |
| `L1-label-d50b723e/L2-add_scene-bc4dab4c/L3-start-7196e7c2/L4-method_unknown_input_2_optional_-e01f8d3c` | 1 | AddScene |
| `L1-label-927fa0e9/L2-aiconference_search-2f4a9d3a/L3-start-7196e7c2/L4-method_unknown_input_1_optional_-fefb130a` | 1 | AIConferenceSearch |
| `L1-label-d50b723e/L2-appointment_registration-a73bf34f/L3-support-0e5b114f/L4-method_unknown_input_3_optional_-fefdb3de` | 1 | AppointmentRegistration |
| `L1-label-ab191893/L2-assign_priority-ab93d9e7/L3-unassigned-b9a5671f/L4-method_unknown_input_2_optional_-e01f8d3c` | 1 | AssignPriority |
| `L1-label-d50b723e/L2-book_hotel-6e10002d/L3-start-7196e7c2/L4-method_unknown_multi_input_6_out-369adebe` | 1 | BookHotel |
| `L1-label-d50b723e/L2-book_ticket-35969c68/L3-support-0e5b114f/L4-method_unknown_multi_input_4_out-d44bf8fb` | 1 | BookTicket |
| `L1-label-d50b723e/L2-buy_train_ticket-16c0cd07/L3-unassigned-b9a5671f/L4-method_unknown_multi_input_6_out-5b332638` | 1 | BuyTrainTicket |
| `L1-label-ad0b9ef9/L2-calculator-46237b3d/L3-finalize-3244ae15/L4-method_unknown_input_1_optional_-fefb130a` | 1 | Calculator |

## Notes

- L1 captures domain, scenario, category, or artifact.
- L2 captures the operation or primary capability.
- L3 captures weak execution role evidence.
- L4 captures IO schema, constraints, examples, or validation details.
